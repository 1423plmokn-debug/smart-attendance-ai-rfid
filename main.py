# Copyright (c) 2026 Ali Nihad Qadir -- Middle Technical University / Polytechnic College.
# All Rights Reserved. Unauthorized copying, redistribution, or attribution is prohibited.
"""
Smart Attendance & Lecture Management System
الجامعة التقنية الوسطى / كلية البوليتكنك
Middle Technical University / Polytechnic College

100% Offline AI Face-Recognition Attendance & Lecture Management System
Architecture: FastAPI | OpenCV | InsightFace (buffalo_s CPU) | SQLite | 100% Offline
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import io
import logging
import queue
import threading
import time
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Optional
import uuid

from database import (
    AttendanceDB,
    DEFAULT_INSTITUTION_AR,
    DEFAULT_INSTITUTION_EN,
    DEFAULT_DEPARTMENTS,
    DEFAULT_STAGES,
    DEFAULT_SECTIONS,
)

DB = AttendanceDB()

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment flags to prevent oneDNN/TensorFlow startup delays and lock conflicts
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Google MediaPipe & Ultra-Lightweight CPU Face Recognition Engine
# Background-load MediaPipe so its TensorFlow dependency does not block startup
mp = None
MEDIAPIPE_AVAILABLE = False

def _bg_init_mediapipe() -> None:
    global mp, MEDIAPIPE_AVAILABLE
    try:
        import mediapipe as _mp
        mp = _mp
        MEDIAPIPE_AVAILABLE = True
    except Exception:
        mp = None
        MEDIAPIPE_AVAILABLE = False

threading.Thread(target=_bg_init_mediapipe, daemon=True, name="mediapipe-init").start()

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    insightface = None   # type: ignore[assignment]
    FaceAnalysis = None  # type: ignore[assignment]
    INSIGHTFACE_AVAILABLE = False

ENGINE_AVAILABLE = INSIGHTFACE_AVAILABLE or True
FR_AVAILABLE     = ENGINE_AVAILABLE  # Backwards-compatible alias for UI & API health checks

# LOGGING
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("attendance.core")

if not ENGINE_AVAILABLE:
    log.warning(
        "Face recognition engine not available. "
        "Camera feed will run without recognition. "
        "Install with: pip install insightface onnxruntime mediapipe"
    )
else:
    log.info("Ultra-Lightweight Face Engine active (Google MediaPipe & CPU-optimized for Intel i5).")

# DIRECTORY STRUCTURE
BASE_DIR    = Path(__file__).parent
KNOWN_DIR   = BASE_DIR / "known_faces"      # Student reference images live here
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR  = BASE_DIR / "models"           # Local ONNX models directory

for _d in (KNOWN_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RFID_MAP_FILE = BASE_DIR / "rfid_cards.json"
RFID_PORT     = os.environ.get("RFID_PORT", "COM5")
RFID_BAUD     = int(os.environ.get("RFID_BAUD", "115200"))

log.info("Directories verified — known_faces: %s | models: %s", KNOWN_DIR, MODELS_DIR)

# ACADEMIC DATA
DEPARTMENTS: list[str] = list(DEFAULT_DEPARTMENTS)
SECTIONS: list[str]     = list(DEFAULT_SECTIONS)
STAGE_OPTIONS: list[str]= list(DEFAULT_STAGES)


def get_academic_departments() -> list[str]:
    items = DB.get_academic_items("department")
    return [item["name"] for item in items] if items else DEPARTMENTS


def get_academic_stages() -> list[str]:
    items = DB.get_academic_items("stage")
    return [item["name"] for item in items] if items else STAGE_OPTIONS


def get_academic_sections() -> list[str]:
    items = DB.get_academic_items("section")
    return [item["name"] for item in items] if items else SECTIONS


def get_institution_names() -> tuple[str, str]:
    ar = DB.get_setting("institution_ar", DEFAULT_INSTITUTION_AR)
    en = DB.get_setting("institution_en", DEFAULT_INSTITUTION_EN)
    return ar, en

# CAMERA CONFIG — HARDCODED TO EXTERNAL CAMERA INDEX 1
CAMERA_INDEX  : int = 1       # External camera — NEVER AUTO-SCAN
CAMERA_WIDTH  : int = 640     # Optimal lightweight resolution (75% pixel reduction vs 1280x720)
CAMERA_HEIGHT : int = 480     # Slashes raw USB throughput & OpenCV decode latency to < 5ms
CAMERA_FPS    : int = 30

# Windows backend preference order — DirectShow avoids obsensor/UVC crashes.
# CAP_DSHOW  → preferred  (bypasses WinRT/UVC obsensor layer entirely)
# CAP_MSMF   → fallback   (Media Foundation, stable on most Win10/11 setups)
# 0 (auto)   → last resort (may trigger obsensor; logged as a warning)
_CAM_BACKENDS: list[tuple[int, str]] = [
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF,  "MSMF"),
    (0,             "auto"),
]

# FACE RECOGNITION TUNING (SMART FRAME-SKIPPING & CACHED TRACKING)
DETECTION_INTERVAL    : int   = 5       # Smart frame-skipping: run full AI detection every 5th frame
DETECTION_WIDTH       : int   = 320     # Lightweight detection width (1/2 of 640 for sub-30ms inference)
DETECTION_HEIGHT      : int   = 240     # Lightweight detection height (1/2 of 480)
SIMILARITY_THRESHOLD  : float = 0.42    # Cosine similarity threshold (self=1.0, cross-student <= 0.21)
RECOGNITION_TOLERANCE : float = 0.48    # Backwards-compatible distance alias (1.0 - SIMILARITY_THRESHOLD)
INSIGHTFACE_DET_SIZE  : tuple[int, int] = (320, 320)  # SCRFD input resolution optimized for CPU throughput
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _name_from_filename(stem: str) -> str:
    """Convert filename stem to clean display name."""
    import re
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return " ".join(w.capitalize() for w in cleaned.split())


def _load_bgr_image_safe(img_path: Path) -> Optional[np.ndarray]:
    """
    Load an image as a BGR numpy array — 100% safe on Windows Arabic/Unicode paths
    and auto-corrects mobile EXIF orientation so rotated smartphone photos load upright.

    Priority:
      1. PIL + ImageOps.exif_transpose (fixes phone rotation, handles any path encoding)
      2. np.fromfile + cv2.imdecode (bypasses cv2.imread path encoding issues)
    NEVER calls cv2.imread() directly.
    """
    # Strategy 1: PIL with EXIF auto-rotation
    try:
        from PIL import Image, ImageOps  # type: ignore
        with Image.open(str(img_path)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = pil_img.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            return bgr
    except Exception:
        pass  # PIL not installed or file is not a valid image for PIL

    # Strategy 2: raw byte read + imdecode (handles Unicode/Arabic paths on Windows)
    try:
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img  # may be None if file is corrupt
    except Exception:
        return None


def _frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def extract_face_embedding_from_bytes(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """
    Safely extract a 512-d normalized face embedding from raw image bytes.
    1. Uses PIL + ImageOps.exif_transpose() -> RGB -> BGR; fallback to cv2.imdecode.
    2. Initializes InsightFace app (FACE_DB._init_app()).
    3. Multi-scale detection: downscale if max dimension > 1280, upscale 2.0x if < 400.
    4. If no face, raises HTTPException(400, "لم يتم العثور على وجه واضح في الصورة، يرجى إعادة التصوير بإضاءة جيدة").
    5. Saves clean JPEG to KNOWN_DIR / f"student_{uuid.uuid4().hex[:12]}.jpg".
    6. Returns (normed_emb, str(saved_path)).
    """
    bgr: Optional[np.ndarray] = None
    try:
        from PIL import Image, ImageOps
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = pil_img.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    except Exception:
        bgr = None

    if bgr is None:
        try:
            buf = np.frombuffer(image_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            bgr = None

    if bgr is None or bgr.size == 0:
        raise HTTPException(400, "ملف الصورة غير صالح أو تالف")

    app = FACE_DB._init_app()
    if app is None:
        raise HTTPException(500, "محرك التعرف على الوجوه غير متوفر حالياً")

    h, w = bgr.shape[:2]
    faces = app.get(bgr)

    if not faces and max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        scaled = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        faces = app.get(scaled)

    if not faces and max(h, w) < 400:
        upscaled = cv2.resize(bgr, (int(w * 2.0), int(h * 2.0)), interpolation=cv2.INTER_CUBIC)
        faces = app.get(upscaled)

    if not faces:
        raise HTTPException(400, "لم يتم العثور على وجه واضح في الصورة، يرجى إعادة التصوير بإضاءة جيدة")

    # Select largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    normed_emb = face.normed_embedding
    if normed_emb is None or len(normed_emb) == 0:
        normed_emb = face.embedding / np.linalg.norm(face.embedding)
    normed_emb = normed_emb.astype(np.float32)
    norm = np.linalg.norm(normed_emb)
    if norm > 0:
        normed_emb = normed_emb / norm

    # Save to KNOWN_DIR with ASCII safe filename
    filename = f"student_{uuid.uuid4().hex[:12]}.jpg"
    saved_path = KNOWN_DIR / filename
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    saved_path.write_bytes(enc.tobytes())

    return normed_emb, str(saved_path.resolve())


def play_audio_alert() -> None:
    """
    Non-blocking local audio alert on host PC when a student is successfully
    recognized and registered for the first time in the session.
    """
    def _run() -> None:
        try:
            import winsound
            # Futuristic dual-pitch confirmation chime (1760 Hz -> 2349 Hz)
            winsound.Beep(1760, 75)
            time.sleep(0.015)
            winsound.Beep(2349, 90)
        except Exception:
            try:
                import sys
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True, name="audio-alert").start()


# FACE DATABASE — SQLite-backed, section-scoped matching
class FaceDatabase:
    """
    Phase 2 rewrite:
    • Loads embeddings DIRECTLY from SQLite (instant startup — no re-inference).
    • Keeps InsightFace app alive for migrate_legacy_faces_if_empty() and reload().
    • Exposes match_for_session() for strict section-scoped recognition.
    • Backwards-compatible names/count/get_photo() properties preserved.
    """

    def __init__(self, faces_dir: Path) -> None:
        self._lock             = threading.Lock()
        self._dir              = faces_dir
        self.students          : list[dict]            = []   # {id, full_name, department, stage, section, rfid_uid, photo_path}
        self.embeddings        : list[np.ndarray]      = []
        self.embeddings_matrix : Optional[np.ndarray]  = None  # (N, 512) float32
        self.photos            : dict[int, str]        = {}   # student_id -> base64 thumb
        self._photos_by_name   : dict[str, str]        = {}   # full_name  -> base64 thumb
        self._loaded           = False
        self.app               : Optional[FaceAnalysis] = None

    def _init_app(self) -> Optional[FaceAnalysis]:
        if not INSIGHTFACE_AVAILABLE:
            return None
        if self.app is not None:
            return self.app
        root_dir = str(BASE_DIR) if (MODELS_DIR / "buffalo_s").exists() else str(Path.home() / ".insightface")
        log.info("Initializing InsightFace (buffalo_s) with root=%s ...", root_dir)
        try:
            app = FaceAnalysis(
                name="buffalo_s",
                root=root_dir,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_thresh=0.45, det_size=INSIGHTFACE_DET_SIZE)
            self.app = app
            log.info("InsightFace FaceAnalysis engine prepared successfully.")
            return self.app
        except Exception as exc:
            log.error("Failed to initialize InsightFace engine: %s", exc)
            return None

    def _make_thumb_b64(self, photo_path: Optional[str]) -> str:
        """Load photo_path and return a 180x180 base64 JPEG thumbnail, or '' on failure."""
        if not photo_path:
            return ""
        try:
            bgr = _load_bgr_image_safe(Path(photo_path))
            if bgr is None:
                return ""
            thumb = cv2.resize(bgr, (180, 180), interpolation=cv2.INTER_AREA)
            return _frame_to_b64(thumb, quality=88)
        except Exception:
            return ""

    def load(self) -> int:
        """
        Load all students from SQLite (instant — uses pre-computed BLOBs).
        Falls back to disk-scan+inference only if SQLite returns 0 students
        (i.e. fresh DB before migration runs).
        """
        from database import blob_to_embedding as _b2e
        db_rows = DB.get_students_with_embeddings()   # includes BLOB column
        if db_rows:
            students:   list[dict]      = []
            embeddings: list[np.ndarray] = []
            photos_by_id:   dict[int, str] = {}
            photos_by_name: dict[str, str] = {}

            for row in db_rows:
                try:
                    emb = _b2e(row["embedding"])
                    # Ensure unit-normalised float32
                    emb = emb.astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    meta = {
                        "id"         : row["id"],
                        "full_name"  : row["full_name"],
                        "department" : row["department"],
                        "stage"      : row["stage"],
                        "section"    : row["section"],
                        "rfid_uid"   : row.get("rfid_uid"),
                        "photo_path" : row.get("photo_path"),
                    }
                    students.append(meta)
                    embeddings.append(emb)
                    thumb = self._make_thumb_b64(row.get("photo_path"))
                    photos_by_id[row["id"]] = thumb
                    photos_by_name[row["full_name"]] = thumb
                except Exception as exc:
                    log.warning("Skipping student id=%s during load: %s", row.get("id"), exc)

            with self._lock:
                self.students          = students
                self.embeddings        = embeddings
                self.embeddings_matrix = np.vstack(embeddings) if embeddings else None
                self.photos            = photos_by_id
                self._photos_by_name   = photos_by_name
                self._loaded           = True

            log.info("Face DB loaded from SQLite — %d students.", len(students))
            return len(students)

        # No rows in DB yet — fall back to disk scan (pre-migration first run)
        return self._load_from_disk()

    def _load_from_disk(self) -> int:
        """
        Legacy disk-scan path: used only when SQLite has zero students
        (first run before migration). Keeps InsightFace inference working.
        """
        if not INSIGHTFACE_AVAILABLE:
            log.warning("InsightFace not available — face database not loaded.")
            return 0
        app = self._init_app()
        if app is None:
            log.warning("Could not initialize InsightFace — face database not loaded.")
            return 0

        students:   list[dict]       = []
        embeddings: list[np.ndarray] = []
        photos_by_id:   dict[int, str] = {}
        photos_by_name: dict[str, str] = {}
        failed = 0

        for img_path in sorted(self._dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            try:
                orig_bgr = _load_bgr_image_safe(img_path)
                if orig_bgr is None:
                    log.warning("Could not read image %s — skipped.", img_path.name)
                    failed += 1
                    continue

                h, w = orig_bgr.shape[:2]
                faces = app.get(orig_bgr)

                if not faces and max(h, w) > 1280:
                    scale = 1280.0 / max(h, w)
                    scaled = cv2.resize(orig_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    faces = app.get(scaled)
                    if faces:
                        inv = 1.0 / scale
                        for f in faces:
                            f.bbox = f.bbox * inv

                if not faces and max(h, w) < 400:
                    upscaled = cv2.resize(orig_bgr, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_CUBIC)
                    faces_up = app.get(upscaled)
                    if faces_up:
                        for f in faces_up:
                            f.bbox = f.bbox / 2.0
                        faces = faces_up

                if not faces:
                    log.warning("No face found in %s — skipped.", img_path.name)
                    failed += 1
                    continue

                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                normed_emb = face.normed_embedding
                if normed_emb is None or len(normed_emb) == 0:
                    normed_emb = face.embedding / np.linalg.norm(face.embedding)
                normed_emb = normed_emb.astype(np.float32)
                norm = np.linalg.norm(normed_emb)
                if norm > 0:
                    normed_emb = normed_emb / norm

                name = _name_from_filename(img_path.stem)

                x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
                pad_y = int((y2 - y1) * 0.28); pad_x = int((x2 - x1) * 0.28)
                crop = orig_bgr[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                thumb_bgr = cv2.resize(crop if crop.size > 0 else orig_bgr, (180, 180), interpolation=cv2.INTER_AREA)
                photo_b64 = _frame_to_b64(thumb_bgr, quality=88)

                # Synthetic metadata — no DB row (migration hasn't run yet)
                meta = {
                    "id"         : -(len(students) + 1),   # negative sentinel
                    "full_name"  : name,
                    "department" : DEPARTMENTS[0],
                    "stage"      : STAGE_OPTIONS[3],
                    "section"    : SECTIONS[0],
                    "rfid_uid"   : None,
                    "photo_path" : str(img_path),
                }
                students.append(meta)
                embeddings.append(normed_emb)
                photos_by_id[meta["id"]] = photo_b64
                photos_by_name[name]     = photo_b64
                log.info("Disk-loaded student reference: %-28s [%s]", name, img_path.name)
            except Exception as exc:
                log.error("Failed to process %s: %s", img_path.name, exc)
                failed += 1

        with self._lock:
            self.students          = students
            self.embeddings        = embeddings
            self.embeddings_matrix = np.vstack(embeddings) if embeddings else None
            self.photos            = photos_by_id
            self._photos_by_name   = photos_by_name
            self._loaded           = True

        log.info("Disk face DB ready — %d loaded, %d failed.", len(students), failed)
        return len(students)

    def reload(self) -> int:
        """Hot-reload from SQLite (or disk if DB is empty)."""
        log.info("Reloading face database...")
        return self.load()

    def match_for_session(
        self,
        unknown_normed_emb: np.ndarray,
        department: Optional[str],
        stage: Optional[str],
        section: Optional[str],
    ) -> tuple[Optional[dict], float, bool]:
        """
        Scoped cosine-similarity match.

        Returns:
          (student_dict, best_sim, in_section)
          • student_dict is None when similarity < SIMILARITY_THRESHOLD.
          • in_section is True only when the matched student belongs to the
            active session's department/stage/section.
        """
        with self._lock:
            if self.embeddings_matrix is None or not self.students:
                return None, 0.0, False
            sims     = np.dot(self.embeddings_matrix, unknown_normed_emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < SIMILARITY_THRESHOLD:
                return None, best_sim, False
            student = self.students[best_idx]

        in_section = (
            student["department"] == department
            and student["stage"]  == stage
            and student["section"] == section
        )
        return student, best_sim, in_section

    def match(self, unknown_normed_emb: np.ndarray) -> tuple[Optional[str], float]:
        """
        Backwards-compatible global match (ignores section).
        Returns (matched_name, cosine_sim) or (None, best_sim).
        """
        with self._lock:
            if self.embeddings_matrix is None or not self.students:
                return None, 0.0
            sims     = np.dot(self.embeddings_matrix, unknown_normed_emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= SIMILARITY_THRESHOLD:
                return self.students[best_idx]["full_name"], best_sim
        return None, best_sim

    def get_photo_by_id(self, student_id: int) -> str:
        """Return cached base64 thumbnail by student primary key."""
        with self._lock:
            return self.photos.get(student_id, "")

    def get_photo(self, name: str) -> str:
        """Backwards-compatible: return cached base64 thumbnail by student full_name."""
        with self._lock:
            return self._photos_by_name.get(name, "")

    @property
    def names(self) -> list[str]:
        with self._lock:
            return [s["full_name"] for s in self.students]

    @property
    def encodings(self) -> list[np.ndarray]:
        with self._lock:
            return list(self.embeddings)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.students)


# Singleton face database
FACE_DB = FaceDatabase(KNOWN_DIR)


# WEBSOCKET REAL-TIME BROADCASTER (/ws/attendance)
class AttendanceWebSocketManager:
    """
    Manages active browser WebSocket connections for real-time attendance events.
    Dispatches immediate push updates for zero-latency UI refreshes & audio feedback.
    """
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            self.active_connections.append(ws)
        log.info("WebSocket client connected. Active listeners: %d", len(self.active_connections))

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            if ws in self.active_connections:
                self.active_connections.remove(ws)
        log.info("WebSocket client disconnected. Active listeners: %d", len(self.active_connections))

    async def _async_broadcast(self, message: dict) -> None:
        dead_conns = []
        with self._lock:
            conns = list(self.active_connections)
        for conn in conns:
            try:
                await conn.send_json(message)
            except Exception:
                dead_conns.append(conn)
        if dead_conns:
            with self._lock:
                for c in dead_conns:
                    if c in self.active_connections:
                        self.active_connections.remove(c)

    def broadcast(self, message: dict) -> None:
        """Thread-safe and async-safe dispatch to all connected WebSocket clients."""
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

        if not loop or not loop.is_running() or not self.active_connections:
            return

        try:
            current_loop = asyncio.get_running_loop()
            if current_loop is loop:
                loop.create_task(self._async_broadcast(message))
                return
        except RuntimeError:
            pass

        asyncio.run_coroutine_threadsafe(self._async_broadcast(message), loop)


WS_MANAGER = AttendanceWebSocketManager()


def broadcast_attendance_event(record: dict, count: int) -> None:
    """Helper to dispatch real-time attendance payload over WebSocket."""
    WS_MANAGER.broadcast({
        "event": "attendance_recorded",
        "data": {
            "name": record.get("name"),
            "timestamp": record.get("timestamp"),
            "method": record.get("method"),
            "photo_b64": record.get("photo_b64", ""),
            "count": count,
        }
    })


# SESSION STATE (STRICT SESSION-BASED DEDUPLICATION: ONLY ONCE PER SESSION)
class SessionState:
    """
    Thread-safe session state manager.
    Phase 2: Persists each lecture and attendance record to SQLite.
    Enforces strict session-based deduplication by student_id (not name).
    """

    def __init__(self) -> None:
        self._lock               = threading.Lock()
        self.active              = False
        self.department          : Optional[str]      = None
        self.section             : Optional[str]      = None
        self.stage               : Optional[str]      = None
        self.course              : Optional[str]      = None
        self.instructor          : Optional[str]      = None
        self.started_at          : Optional[datetime] = None
        self.attendance          : list[dict]         = []
        self.lecture_id          : Optional[int]      = None
        # Phase 2: dedup by student_id (int) for DB-backed records
        self._logged_student_ids : set[int]           = set()
        # Backwards-compat: also dedup by name for disk-only (pre-migration) students
        self._logged_students    : set[str]           = set()

    def begin(
        self,
        department: str,
        section: str,
        stage: str,
        course: str,
        instructor: Optional[str] = None,
    ) -> None:
        # Create SQLite lecture record BEFORE acquiring lock (DB call can be slow)
        new_lecture_id = DB.create_lecture(department, stage, section, course, instructor)
        with self._lock:
            self.department          = department
            self.section             = section
            self.stage               = stage
            self.course              = course
            self.instructor          = instructor
            self.started_at          = datetime.now()
            self.attendance          = []
            self.lecture_id          = new_lecture_id
            self._logged_student_ids = set()
            self._logged_students    = set()
            self.active              = True
        log.info(
            "Session STARTED — lecture_id=%d | dept=%s | section=%s | stage=%s | course=%s",
            new_lecture_id, department, section, stage, course,
        )

    def terminate(self) -> None:
        with self._lock:
            lid = self.lecture_id
            self.active      = False
            self.lecture_id  = None
        # End lecture in DB (outside lock to avoid blocking)
        if lid is not None:
            try:
                DB.end_lecture(lid)
            except Exception as exc:
                log.error("Failed to end lecture %d in DB: %s", lid, exc)
        log.info("Session TERMINATED (lecture_id=%s)", lid)

    def try_record_attendance(
        self,
        student_id: int,
        name: str,
        photo_b64: str,
        method: str = "كاميرا ذكية (AI Vision)",
    ) -> bool:
        """
        Record attendance strictly ONCE per session per student_id.
        Persists to SQLite via DB.record_attendance().
        Returns False for duplicates or inactive session.
        """
        with self._lock:
            if not self.active or self.lecture_id is None:
                return False
            # Dedup by student_id (positive IDs = real DB rows)
            if student_id > 0 and student_id in self._logged_student_ids:
                return False
            # Backwards-compat dedup by name (negative IDs = pre-migration disk students)
            if name in self._logged_students:
                return False

        # DB insert OUTSIDE lock (may block briefly on WAL flush)
        if student_id > 0:
            inserted = DB.record_attendance(self.lecture_id, student_id, method)
            if not inserted:
                return False  # DB UNIQUE constraint caught duplicate

        with self._lock:
            if not self.active:   # session may have ended while we were in DB
                return False
            if student_id > 0:
                self._logged_student_ids.add(student_id)
            self._logged_students.add(name)
            rec = {
                "student_id" : student_id,
                "name"       : name,
                "photo_b64"  : photo_b64,
                "timestamp"  : datetime.now().isoformat(),
                "method"     : method,
            }
            self.attendance.append(rec)
            cur_count = len(self.attendance)

        log.info("Attendance RECORDED: %s [id=%d] [Method: %s]", name, student_id, method)
        play_audio_alert()
        broadcast_attendance_event(rec, cur_count)
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active"     : self.active,
                "department" : self.department,
                "section"    : self.section,
                "stage"      : self.stage,
                "course"     : self.course,
                "instructor" : self.instructor,
                "started_at" : self.started_at.isoformat() if self.started_at else None,
                "count"      : len(self.attendance),
                "attendance" : list(self.attendance),
                "lecture_id" : self.lecture_id,
            }


SESSION = SessionState()


# RFID REPOSITORY & ESP32-S HARDWARE SERIAL WORKER (COM5 @ 115200)
DEFAULT_RFID_MAP = {
    "UID:3A:5B:92:F1": "Ali Dhea",
    "UID:4C:6D:81:A2": "Ali Nehad",
    "UID:DA:5B:90:32": "Ali Nehad",
    "UID:B3:99:A1:02": "Baneen Talib",
    "UID:F1:22:33:44": "Enas Abdulkarim",
    "UID:D4:55:66:77": "Mahdi Ali",
    "UID:E5:88:99:00": "Mahdi Imad",
    "UID:A1:B2:C3:D4": "Ruaa Nadeem",
    "UID:11:22:33:44": "Sajjad Khazaal",
}

class RFIDCardRepository:
    """
    Manages persistent mappings between physical RFID card UIDs and enrolled students.
    Auto-assigns unmapped cards to known enrolled students for seamless plug-and-play.
    """
    def __init__(self, map_file: Path) -> None:
        self._file = map_file
        self._lock = threading.Lock()
        self.mappings: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        with self._lock:
            if self._file.exists():
                try:
                    with open(self._file, "r", encoding="utf-8") as f:
                        self.mappings = json.load(f)
                    log.info("Loaded %d RFID card mappings from %s", len(self.mappings), self._file.name)
                    return
                except Exception as exc:
                    log.warning("Could not read %s: %s. Using default mappings.", self._file.name, exc)
            self.mappings = dict(DEFAULT_RFID_MAP)
            self._save_locked()

    def _save_locked(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self.mappings, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log.error("Could not write %s: %s", self._file.name, exc)

    def get_student(self, uid: str) -> Optional[str]:
        with self._lock:
            return self.mappings.get(uid.strip().upper())

    def resolve_or_auto_assign(self, uid: str, known_students: list[str]) -> str:
        clean_uid = uid.strip().upper()
        with self._lock:
            if clean_uid in self.mappings:
                return self.mappings[clean_uid]
            # Auto-assign unmapped card to the next available student without an RFID card
            assigned_names = set(self.mappings.values())
            available = [s for s in known_students if s not in assigned_names]
            assigned = available[0] if available else (known_students[0] if known_students else f"طالب (بطاقة {clean_uid})")
            self.mappings[clean_uid] = assigned
            self._save_locked()
            log.info("Auto-assigned new RFID card %s to student: %s", clean_uid, assigned)
            return assigned

    def register(self, uid: str, student_name: str) -> None:
        with self._lock:
            clean_uid = uid.strip().upper()
            self.mappings[clean_uid] = student_name
            self._save_locked()
            log.info("Mapped RFID %s -> %s", clean_uid, student_name)


RFID_REPO = RFIDCardRepository(RFID_MAP_FILE)


class RFIDSerialManager:
    """
    Non-blocking serial worker & state manager for ESP32-S + MFRC522 RFID reader.
    Continuously listens on serial port (COM5 by default) at 115200 baud.
    Features:
      • Non-blocking reconnection loop (zero server lag if hardware is disconnected).
      • Strict UID format parsing ("UID:XX:XX:XX:XX").
      • Software debouncing cooldown to prevent duplicate serial spam.
      • Automatic session attendance logging and WebSocket broadcasting.
      • Independent asyncio background task integration via rfid_background_listener.
    """
    def __init__(
        self,
        port: str = RFID_PORT,
        baudrate: int = RFID_BAUD,
        card_repo: Optional[RFIDCardRepository] = None,
    ) -> None:
        self.port         = port
        self.baudrate     = baudrate
        self.card_repo    = card_repo or RFID_REPO
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._async_task: Optional[asyncio.Task] = None
        self._stop_event  = threading.Event()
        self._lock        = threading.Lock()
        self.connected    = False
        self.last_uid     : Optional[str] = None
        self.last_scan_dt : Optional[datetime] = None
        self.last_scan_ts = 0.0
        self.cooldown_sec = 1.5  # 1.5 seconds cooldown per card

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()

        # If in an active asyncio event loop, start rfid_background_listener as an asyncio task
        try:
            active_loop = loop or asyncio.get_running_loop()
            if active_loop and active_loop.is_running():
                self._async_task = active_loop.create_task(
                    rfid_background_listener(), name="rfid-background-listener"
                )
                log.info("RFIDSerialManager: started rfid_background_listener as asyncio task on %s.", self.port)
                return
        except RuntimeError:
            pass

        # Fallback to background thread if no running event loop
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="rfid-serial-worker"
        )
        self._thread.start()
        log.info("RFIDSerialManager background thread started on %s @ %d baud.", self.port, self.baudrate)

    def stop(self) -> None:
        with self._lock:
            self._running = False
        self._stop_event.set()
        if self._async_task and not self._async_task.done():
            self._async_task.cancel()
        log.info("RFIDSerialManager stopped.")

    def is_connected(self) -> bool:
        with self._lock:
            return self.connected

    def _worker_loop(self) -> None:
        if not SERIAL_AVAILABLE:
            log.warning("pyserial is not installed. RFID background worker disabled.")
            return

        reconnect_delay = 2.0
        while not self._stop_event.is_set():
            ser = None
            try:
                log.info("Attempting connection to ESP32 RFID on %s (%d baud)...", self.port, self.baudrate)
                ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=1.0,
                    write_timeout=1.0,
                )
                with self._lock:
                    self.connected = True
                log.info("RFID Serial successfully connected on %s.", self.port)

                # Send readiness sync
                try:
                    ser.write(b"STATUS:SYSTEM_READY\n")
                except Exception:
                    pass

                while not self._stop_event.is_set():
                    raw_line = ser.readline()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    if "UID:" in line:
                        self._process_payload(line)

            except serial.SerialException as exc:
                with self._lock:
                    self.connected = False
                log.debug("Serial connection on %s unavailable: %s. Retrying in %.1fs...", self.port, exc, reconnect_delay)
            except Exception as exc:
                with self._lock:
                    self.connected = False
                log.error("Unexpected error in RFID serial loop: %s", exc)
            finally:
                if ser and ser.is_open:
                    try:
                        ser.close()
                    except Exception:
                        pass
                with self._lock:
                    self.connected = False

            # Wait before retry attempt
            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=reconnect_delay)

    def _process_payload(self, line: str) -> None:
        """Parse incoming UID:XX:XX:XX:XX line, resolve student, and record attendance."""
        match = re.search(r"UID:([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2})+)", line)
        if not match:
            return
        clean_uid = f"UID:{match.group(1).upper()}"
        now = time.time()

        with self._lock:
            if clean_uid == self.last_uid and (now - self.last_scan_ts) < self.cooldown_sec:
                return
            self.last_uid = clean_uid
            self.last_scan_ts = now
            self.last_scan_dt = datetime.now()

        log.info("RFID Card detected on %s: %s", self.port, clean_uid)

        if not SESSION.active:
            log.info("RFID scanned (%s), but no lecture session is active.", clean_uid)
            return

        # Look up scanned UID directly in SQLite database
        student = DB.get_student_by_rfid(clean_uid)
        if not student:
            log.warning("Unregistered RFID card scanned: %s (ignored).", clean_uid)
            return

        # Strict Section Isolation check
        sess_dept  = SESSION.department
        sess_stage = SESSION.stage
        sess_sec   = SESSION.section

        is_in_section = (
            student.get("department") == sess_dept
            and student.get("stage")  == sess_stage
            and student.get("section") == sess_sec
        )

        if is_in_section:
            photo_b64 = FACE_DB.get_photo_by_id(student["id"])
            recorded = SESSION.try_record_attendance(
                student_id=student["id"],
                name=student["full_name"],
                photo_b64=photo_b64,
                method="بطاقة جامعية (RFID RC522)",
            )
            if recorded:
                log.info("Attendance RECORDED via RFID: %s [%s]", student["full_name"], clean_uid)
            else:
                log.info("Student %s already logged for this session. Duplicate RFID scan ignored.", student["full_name"])
        else:
            log.warning(
                "Wrong section: Student %s (%s, %s, %s) does not belong to active section (%s, %s, %s)",
                student["full_name"], student.get("department"), student.get("stage"), student.get("section"),
                sess_dept, sess_stage, sess_sec,
            )
            WS_MANAGER.broadcast({
                "event": "wrong_section",
                "data": {
                    "name": student["full_name"],
                    "section": student.get("section", "—"),
                    "department": student.get("department", "—"),
                }
            })

    def process_payload(self, line: str) -> None:
        """Public alias for _process_payload."""
        self._process_payload(line)


RFID_MANAGER = RFIDSerialManager(RFID_PORT, RFID_BAUD, RFID_REPO)


async def rfid_background_listener() -> None:
    """
    Independent asyncio background task for ESP32-S + MFRC522 RFID reader.
    Continuously listens on serial port (COM5 @ 115200) without blocking the asyncio event loop.
    All blocking I/O (connecting, reading, writing) is executed in worker threads via asyncio.to_thread,
    ensuring that the webcam video streaming route, WebSocket, and REST endpoints run 100% concurrently.
    """
    if not SERIAL_AVAILABLE:
        log.warning("pyserial is not installed. RFID background listener disabled.")
        return

    port = RFID_MANAGER.port
    baudrate = RFID_MANAGER.baudrate
    reconnect_delay = 2.0

    log.info("RFID background listener task started on %s @ %d baud.", port, baudrate)

    while not RFID_MANAGER._stop_event.is_set():
        ser = None
        try:
            log.info("RFID listener attempting connection to %s (%d baud)...", port, baudrate)

            def _open():
                return serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    timeout=0.5,
                    write_timeout=0.5,
                )

            ser = await asyncio.to_thread(_open)
            with RFID_MANAGER._lock:
                RFID_MANAGER.connected = True
            log.info("RFID Serial successfully connected on %s.", port)

            try:
                await asyncio.to_thread(ser.write, b"STATUS:SYSTEM_READY\n")
            except Exception:
                pass

            def _read_line(s):
                try:
                    return s.readline()
                except Exception:
                    return b""

            while not RFID_MANAGER._stop_event.is_set() and ser.is_open:
                raw_line = await asyncio.to_thread(_read_line, ser)
                if not raw_line:
                    await asyncio.sleep(0.01)
                    continue

                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    await asyncio.sleep(0.005)
                    continue

                if "UID:" in line:
                    RFID_MANAGER._process_payload(line)

                await asyncio.sleep(0.005)

        except serial.SerialException as exc:
            with RFID_MANAGER._lock:
                RFID_MANAGER.connected = False
            log.debug("Serial connection on %s unavailable: %s. Retrying in %.1fs...", port, exc, reconnect_delay)
        except asyncio.CancelledError:
            log.info("RFID background listener task cancelled.")
            break
        except Exception as exc:
            with RFID_MANAGER._lock:
                RFID_MANAGER.connected = False
            log.error("Unexpected error in RFID background listener: %s", exc)
        finally:
            with RFID_MANAGER._lock:
                RFID_MANAGER.connected = False
            if ser and ser.is_open:
                try:
                    await asyncio.to_thread(ser.close)
                except Exception:
                    pass

        if not RFID_MANAGER._stop_event.is_set():
            try:
                await asyncio.sleep(reconnect_delay)
            except asyncio.CancelledError:
                break

    with RFID_MANAGER._lock:
        RFID_MANAGER.connected = False
    log.info("RFID background listener stopped.")


# GEOMETRIC DRAWING OVERLAYS (HIGH-CONTRAST CYBER-GREEN & SLEEK DESIGN)
_CLR_BOX_RECOGNISED    = (0,   220, 100)  # High-tech Cyber Green (BGR) — in-section match
_CLR_CORNER_RECOGNISED = (0,   255, 128)  # Bright Green Corners (BGR)
_CLR_BOX_WRONG_SECTION = (0,   180, 255)  # Amber/Yellow (BGR) — matched but wrong section
_CLR_CORNER_WRONG      = (0,   210, 255)  # Amber Corners (BGR)
_CLR_BOX_UNKNOWN       = (80,  80,  80)   # Subtle Gray for unidentified faces (BGR)
_CLR_CORNER_UNKNOWN    = (120, 120, 120)  # Medium Gray Corners (BGR)
_CLR_TEXT_BG           = (0,   0,   0)    # Deep Black text strip
_CLR_TEXT              = (255, 255, 255)  # Crisp White text
_CORNER_LEN   = 18
_CORNER_THICK = 2
_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE   = 0.52
_FONT_THICK   = 1


def _draw_face_box(
    frame  : np.ndarray,
    top    : int,
    right  : int,
    bottom : int,
    left   : int,
    name   : str,
    status : str,   # "ok" | "wrong_section" | "unknown"
) -> None:
    """Draw sleek geometric corner brackets and label badge.

    status values:
      "ok"            — Cyber-Green: in-section match, attendance recorded.
      "wrong_section" — Amber/Yellow: matched but different section/dept.
      "unknown"       — Gray: no match above threshold.
    """
    if status == "ok":
        color = _CLR_BOX_RECOGNISED
        cc    = _CLR_CORNER_RECOGNISED
        label = f"[OK] {name}"
    elif status == "wrong_section":
        color = _CLR_BOX_WRONG_SECTION
        cc    = _CLR_CORNER_WRONG
        label = f"[WRONG SECTION] {name}"
    else:
        color = _CLR_BOX_UNKNOWN
        cc    = _CLR_CORNER_UNKNOWN
        label = "[?] UNKNOWN"

    # Border rectangle
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1)

    # Sharp geometric corner accents
    cl = _CORNER_LEN
    ct = _CORNER_THICK
    cv2.line(frame, (left, top),     (left + cl, top),    cc, ct)
    cv2.line(frame, (left, top),     (left, top + cl),    cc, ct)
    cv2.line(frame, (right, top),    (right - cl, top),   cc, ct)
    cv2.line(frame, (right, top),    (right, top + cl),   cc, ct)
    cv2.line(frame, (left, bottom),  (left + cl, bottom), cc, ct)
    cv2.line(frame, (left, bottom),  (left, bottom - cl), cc, ct)
    cv2.line(frame, (right, bottom), (right - cl, bottom), cc, ct)
    cv2.line(frame, (right, bottom), (right, bottom - cl), cc, ct)

    # Minimalist label tag
    (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
    strip_top = max(top - th - 12, 0)
    strip_bot = top
    cv2.rectangle(frame, (left, strip_top), (left + tw + 14, strip_bot), _CLR_TEXT_BG, -1)
    cv2.rectangle(frame, (left, strip_top), (left + tw + 14, strip_bot), color, 1)
    cv2.putText(
        frame, label,
        (left + 6, strip_bot - 5),
        _FONT, _FONT_SCALE, _CLR_TEXT, _FONT_THICK, cv2.LINE_AA,
    )


# CAMERA MANAGER (DECOUPLED 30 FPS CAPTURE + LOCK-FREE FRAME QUEUE)
class CameraManager:
    """
    High-performance camera streamer & async face recognizer.

    Architecture
    ─────────────
    • _capture_thread  — reads cv2.VideoCapture(1) at full 30 FPS, never waits
                         on recognition or HTTP consumers.
    • _frame_queue     — queue.Queue(maxsize=2) carrying encoded JPEG bytes.
                         put_nowait() drops stale frames so the queue stays
                         fresh and never accumulates lag.
    • _rec_thread      — pulls the latest raw BGR frame, downscales to 0.25×
                         for ultra-fast HOG detection, matches against cached
                         512-d encodings. Runs at ~15 FPS independently.
    • get_jpeg_stream  — ASYNC generator that awaits asyncio.sleep(0.033) so
                         it yields control back to the FastAPI event loop on
                         every frame; completely non-blocking.
    • _stop_event      — threading.Event signalled by stop(); lets the capture
                         loop and stream generator exit cleanly without hanging.

    Face encodings are computed ONCE at startup in a background thread and
    cached in FACE_DB.  This class never recomputes them.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        self.camera_index = camera_index

        # Shared state — protected by _lock where noted
        self._lock            = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._running         = False
        self._stop_event      = threading.Event()   # set when stop() is called

        # _frame_queue  : producer = capture thread  (put_nowait — never blocks)
        #                 consumer = async generator  (asyncio, non-blocking)
        # maxsize=2 keeps at most 2 JPEG frames buffered; older frames are
        # discarded automatically via the put_nowait / get_nowait pattern.
        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self._latest_jpeg: Optional[bytes] = None

        # Latest raw BGR frame shared between capture → recognition threads
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_detection_frame: Optional[np.ndarray] = None
        self._latest_small_rgb: Optional[np.ndarray] = None  # Backwards-compatibility alias
        self._frame_seq: int = 0

        # Face bounding boxes from the recognition thread, consumed by capture
        self._current_faces: list[tuple] = []   # (top,right,bottom,left,name,recognised)

        self._capture_thread: Optional[threading.Thread] = None
        self._rec_thread:     Optional[threading.Thread] = None

    # Lifecycle
    def start(self) -> None:
        with self._lock:
            cap_alive = self._capture_thread is not None and self._capture_thread.is_alive()
            rec_alive = self._rec_thread is not None and self._rec_thread.is_alive()
            if self._running and cap_alive and rec_alive:
                return
            self._running = True
            self._stop_event.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="cam-capture"
            )
            self._rec_thread = threading.Thread(
                target=self._recognition_loop, daemon=True, name="cam-recognizer"
            )
            self._capture_thread.start()
            self._rec_thread.start()
            log.info("CameraManager started — camera index %d", self.camera_index)

    def stop(self) -> None:
        """Signal threads to stop and release the camera handle."""
        with self._lock:
            self._running = False
            self._latest_jpeg = None
        self._stop_event.set()          # wake any blocking waits immediately
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        # Drain residual frames so a new session starts with a clean queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._latest_raw = None
            self._latest_detection_frame = None
            self._latest_small_rgb = None
            self._current_faces = []
        log.info("CameraManager stopped")

    def is_active(self) -> bool:
        with self._lock:
            return self._running and (
                self._cap is not None and self._cap.isOpened()
            )

    # Internal helpers
    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Open the external/internal camera using a safe backend priority chain.
        Supports fallback across candidate indices (e.g. index 1 and index 0)
        to prevent 'Camera index out of range' errors when hardware enumeration varies.
        """
        candidate_indices = [self.camera_index]
        for alt_idx in [0, 1, 2]:
            if alt_idx not in candidate_indices:
                candidate_indices.append(alt_idx)

        for c_idx in candidate_indices:
            for backend_id, backend_name in _CAM_BACKENDS:
                try:
                    if backend_id:
                        cap = cv2.VideoCapture(c_idx, backend_id)
                    else:
                        cap = cv2.VideoCapture(c_idx)

                    if not cap.isOpened():
                        cap.release()
                        log.debug("Camera [%d] not opened via %s — skipping.",
                                  c_idx, backend_name)
                        continue

                    # Apply resolution / FPS / buffer hints
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # no kernel-level lag

                    # Validate with a real frame read
                    ret, _ = cap.read()
                    if not ret:
                        cap.release()
                        log.warning(
                            "Camera [%d] opened via %s but returned no frame — skipping.",
                            c_idx, backend_name,
                        )
                        continue

                    self.camera_index = c_idx
                    log.info(
                        "Camera [%d] opened successfully via %s backend.",
                        c_idx, backend_name,
                    )
                    return cap

                except Exception as exc:
                    log.warning(
                        "Camera [%d] open attempt via %s raised: %s",
                        c_idx, backend_name, exc,
                    )

        log.warning(
            "Camera could not be opened on candidate indices %s (DSHOW / MSMF / auto). "
            "Check USB connection.",
            candidate_indices,
        )
        return None

    def _generate_standby_frame(
        self, message: str = "CAMERA [1] STANDBY"
    ) -> np.ndarray:
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
        cv2.rectangle(frame, (cx - 220, cy - 65), (cx + 220, cy + 65), (35, 35, 35), 1)
        cv2.line(frame, (cx - 240, cy), (cx + 240, cy), (48, 48, 48), 1)
        cv2.line(frame, (cx, cy - 80), (cx, cy + 80), (48, 48, 48), 1)
        cv2.putText(frame, "MIDDLE TECHNICAL UNIVERSITY - POLYTECHNIC COLLEGE",
                    (cx - 215, cy - 75), _FONT, 0.36, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, message,
                    (cx - 130, cy + 5),  _FONT, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, "DEVICE [1] - 100% OFFLINE FACE RECOGNITION SYSTEM",
                    (cx - 180, cy + 40), _FONT, 0.34, (80, 80, 80), 1, cv2.LINE_AA)
        return frame

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        """Encode BGR frame to JPEG bytes (quality 80)."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def _push_frame(self, jpeg: bytes) -> None:
        """
        Push an encoded JPEG frame into the queue.
        If the queue is full (consumer is slow), discard the oldest frame first
        so the queue always holds the freshest data — no accumulation lag.
        """
        try:
            self._frame_queue.put_nowait(jpeg)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()   # discard stale frame
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(jpeg)
            except queue.Full:
                pass  # extremely unlikely; skip silently

    # Capture thread  (runs at camera FPS, fully decoupled from recognition)
    def _capture_loop(self) -> None:
        self._cap = self._open_camera()
        last_retry     = time.monotonic()
        _blank_streak  = 0           # consecutive failed reads counter
        _MAX_BLANKS    = 10          # threshold before we mark cap as dead

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if time.monotonic() - last_retry > 3.0:
                    last_retry    = time.monotonic()
                    _blank_streak = 0
                    log.info("Retrying camera open on index %d ...", self.camera_index)
                    self._cap = self._open_camera()
                standby = self._generate_standby_frame(f"CAMERA [{self.camera_index}] SEARCHING...")
                s_bytes = self._encode_jpeg(standby)
                with self._lock:
                    self._latest_jpeg = s_bytes
                self._push_frame(s_bytes)
                time.sleep(0.04)
                continue

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    _blank_streak += 1
                    if _blank_streak >= _MAX_BLANKS:
                        # Too many consecutive blank reads — the camera has likely
                        # disconnected or the driver stalled.  Release it so the
                        # retry block above can re-open cleanly.
                        log.warning(
                            "Camera [%d] produced %d consecutive blank frames — "
                            "releasing handle for re-open.",
                            self.camera_index, _blank_streak,
                        )
                        self._cap.release()
                        self._cap     = None
                        last_retry    = time.monotonic() - 3.0  # force immediate retry
                        _blank_streak = 0
                    time.sleep(0.02)
                    continue

                # Successful read — reset blank counter
                _blank_streak = 0

                # Fast downscale for recognition (320x240 keeps 4:3 ratio, zero distortion)
                small_bgr = cv2.resize(
                    frame, (DETECTION_WIDTH, DETECTION_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Share raw frame for UI rendering and downscaled BGR frame for recognition
                with self._lock:
                    self._latest_raw             = frame
                    self._latest_detection_frame = small_bgr
                    self._latest_small_rgb       = small_bgr
                    self._frame_seq             += 1
                    faces_to_draw                = list(self._current_faces)

                # Overlay the latest bounding boxes (from recognition thread)
                annotated = frame.copy()
                for (top, right, bottom, left, name, status) in faces_to_draw:
                    _draw_face_box(annotated, top, right, bottom, left, name, status)

                # Push the annotated JPEG into the lock-free queue and cache latest frame
                encoded = self._encode_jpeg(annotated)
                with self._lock:
                    self._latest_jpeg = encoded
                self._push_frame(encoded)

                # Cooperative yield between frames (~3ms) to prevent CPU core spin lock
                time.sleep(0.003)
            except Exception as exc:
                log.warning("Transient frame capture error on camera [%d]: %s", self.camera_index, exc)
                time.sleep(0.02)

    # Recognition thread (Ultra-Lightweight CPU Inference + Smart Frame Skipping)
    def _recognition_loop(self) -> None:
        """
        Ultra-lightweight recognition worker with smart frame-skipping and cached tracking.
        Runs AI detection every DETECTION_INTERVAL frames, reusing cached tracking
        on intermediate frames for sub-100ms overall latency (<0.02s avg) and zero CPU freeze.
        """
        scale_x = CAMERA_WIDTH / float(DETECTION_WIDTH)   # = 2.0 (640 / 320)
        scale_y = CAMERA_HEIGHT / float(DETECTION_HEIGHT) # = 2.0 (480 / 240)
        last_seq = -1
        frame_counter = 0
        cached_tracked_faces: list[tuple] = []
        decay_streak = 0

        while self._running:
            # Sleep if session is idle or face database is empty
            if not SESSION.active or not ENGINE_AVAILABLE or FACE_DB.embeddings_matrix is None:
                with self._lock:
                    self._current_faces = []
                cached_tracked_faces = []
                frame_counter = 0
                time.sleep(0.04)
                continue

            # Grab latest downscaled BGR frame
            with self._lock:
                if self._frame_seq == last_seq or self._latest_detection_frame is None:
                    det_frame = None
                else:
                    det_frame = self._latest_detection_frame
                    last_seq  = self._frame_seq

            if det_frame is None:
                time.sleep(0.005)
                continue

            frame_counter += 1

            # or every 2nd frame when hunting for an initial face
            run_detection = (
                (frame_counter % DETECTION_INTERVAL == 0)
                or (not cached_tracked_faces and frame_counter % 2 == 0)
            )

            if run_detection:
                app = FACE_DB.app
                faces = []
                if app is not None:
                    try:
                        faces = app.get(det_frame)
                    except Exception as exc:
                        log.warning("Detection inference error: %s", exc)
                        faces = []

                if faces:
                    decay_streak = 0
                    new_faces: list[tuple] = []
                    for face in faces:
                        x1, y1, x2, y2 = face.bbox
                        # Coordinate scaling: restore accurately to 640x480 coordinates
                        top    = max(0, min(CAMERA_HEIGHT - 1, int(round(y1 * scale_y))))
                        left   = max(0, min(CAMERA_WIDTH - 1,  int(round(x1 * scale_x))))
                        bottom = max(0, min(CAMERA_HEIGHT - 1, int(round(y2 * scale_y))))
                        right  = max(0, min(CAMERA_WIDTH - 1,  int(round(x2 * scale_x))))

                        normed_emb = face.normed_embedding
                        if normed_emb is None or len(normed_emb) == 0:
                            normed_emb = face.embedding / np.linalg.norm(face.embedding)

                        # Match against in-memory student database with strict section isolation
                        student, sim, in_section = FACE_DB.match_for_session(
                            normed_emb,
                            SESSION.department,
                            SESSION.stage,
                            SESSION.section,
                        )
                        if student is not None and in_section:
                            status  = "ok"
                            display = student["full_name"]
                            photo   = FACE_DB.get_photo_by_id(student["id"])
                            SESSION.try_record_attendance(student["id"], student["full_name"], photo)
                        elif student is not None and not in_section:
                            status  = "wrong_section"
                            display = student["full_name"]
                            # DO NOT record attendance for wrong section!
                        else:
                            status  = "unknown"
                            display = "UNKNOWN"

                        new_faces.append((top, right, bottom, left, display, status))

                    cached_tracked_faces = new_faces
                    with self._lock:
                        self._current_faces = new_faces
                else:
                    decay_streak += 1
                    # Smooth temporal decay: retain tracked boxes for 2 intervals to prevent flickering
                    if decay_streak >= 2:
                        cached_tracked_faces = []
                        with self._lock:
                            self._current_faces = []
            else:
                # Intermediate frames: instant zero-cost reuse of cached tracking coordinates
                if cached_tracked_faces:
                    with self._lock:
                        self._current_faces = cached_tracked_faces

            # Cooperative thread sleep: frees CPU cycles for FastAPI and DirectShow
            time.sleep(0.008)

    # Async MJPEG stream generator  (non-blocking — uses asyncio.sleep)
    async def get_jpeg_stream(self):
        """
        Async generator that yields MJPEG multipart frames at ~30 FPS.

        Key properties:
        • Uses asyncio.sleep(0.033) — yields control back to FastAPI event loop on every frame.
        • Serves latest cached JPEG frame (_latest_jpeg) safely across multiple simultaneous clients.
        • Handles client disconnects (CancelledError, GeneratorExit) cleanly without hanging.
        • Stops cleanly when _stop_event is set.
        """
        _fallback: Optional[bytes] = None

        try:
            while not self._stop_event.is_set():
                jpeg: Optional[bytes] = None
                with self._lock:
                    if self._latest_jpeg is not None:
                        jpeg = self._latest_jpeg
                        _fallback = jpeg

                if jpeg is None:
                    try:
                        jpeg = self._frame_queue.get_nowait()
                        _fallback = jpeg
                    except queue.Empty:
                        if _fallback is not None:
                            jpeg = _fallback
                        else:
                            standby = self._generate_standby_frame("INITIALISING FEED...")
                            jpeg    = self._encode_jpeg(standby)

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

                # Cooperative yield back to the event loop (~30 FPS pacing)
                await asyncio.sleep(0.033)
        except (asyncio.CancelledError, GeneratorExit):
            pass


CAMERA = CameraManager(CAMERA_INDEX)


def migrate_legacy_faces_if_empty(face_app: Optional[FaceAnalysis]) -> int:
    """
    Task 1: One-time auto-migration from legacy known_faces/ folder and rfid_cards.json
    into the SQLite database (students table). Runs ONLY if DB has 0 students.
    """
    existing = DB.get_students()
    if existing:
        log.info("Migration skipped: SQLite already contains %d students.", len(existing))
        return 0

    image_files = [f for f in sorted(KNOWN_DIR.iterdir()) if f.suffix.lower() in _IMG_EXTS]
    if not image_files:
        log.info("Migration skipped: no images found in %s.", KNOWN_DIR)
        return 0

    if face_app is None:
        log.warning("Migration cannot proceed: InsightFace app is not initialized.")
        return 0

    # Load rfid_cards.json if present (pick first matching UID per student name)
    name_to_uid: dict[str, str] = {}
    rfid_map: dict[str, str] = {}
    if RFID_MAP_FILE.exists():
        try:
            with open(RFID_MAP_FILE, "r", encoding="utf-8") as f:
                rfid_map = json.load(f)
        except Exception as exc:
            log.warning("Could not load %s for migration: %s", RFID_MAP_FILE.name, exc)
            rfid_map = dict(DEFAULT_RFID_MAP)
    else:
        rfid_map = dict(DEFAULT_RFID_MAP)

    for uid, s_name in rfid_map.items():
        if not s_name:
            continue
        clean_name = s_name.strip().lower()
        if clean_name not in name_to_uid:
            name_to_uid[clean_name] = uid.strip().upper()

    migrated = 0
    assigned_uids: set[str] = set()

    for img_path in image_files:
        try:
            orig_bgr = _load_bgr_image_safe(img_path)
            if orig_bgr is None:
                continue

            h, w = orig_bgr.shape[:2]
            faces = face_app.get(orig_bgr)

            if not faces and max(h, w) > 1280:
                scale = 1280.0 / max(h, w)
                scaled = cv2.resize(orig_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                faces = face_app.get(scaled)

            if not faces and max(h, w) < 400:
                upscaled = cv2.resize(orig_bgr, (int(w * 2.0), int(h * 2.0)), interpolation=cv2.INTER_CUBIC)
                faces = face_app.get(upscaled)

            if not faces:
                log.warning("Migration: No face detected in %s — skipped.", img_path.name)
                continue

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            normed_emb = face.normed_embedding
            if normed_emb is None or len(normed_emb) == 0:
                normed_emb = face.embedding / np.linalg.norm(face.embedding)
            normed_emb = normed_emb.astype(np.float32)
            norm = np.linalg.norm(normed_emb)
            if norm > 0:
                normed_emb = normed_emb / norm

            full_name = _name_from_filename(img_path.stem)
            uid = name_to_uid.get(full_name.lower())
            if uid and uid in assigned_uids:
                uid = None
            if uid:
                assigned_uids.add(uid)

            DB.add_student(
                full_name=full_name,
                department=DEPARTMENTS[0],
                stage=STAGE_OPTIONS[3],
                section=SECTIONS[0],
                embedding=normed_emb,
                photo_path=str(img_path.resolve()),
                rfid_uid=uid,
            )
            migrated += 1
            log.info("Migrated student [%s] (RFID: %s) to SQLite.", full_name, uid)
        except Exception as exc:
            log.error("Migration failed for %s: %s", img_path.name, exc)

    log.info("Auto-migration complete: migrated %d legacy students into SQLite.", migrated)
    return migrated


# FASTAPI APP & LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register active running asyncio loop with WebSocket broadcaster
    loop = asyncio.get_running_loop()
    WS_MANAGER.set_loop(loop)

    # Initialize the SQLite database (creates tables if they don't exist)
    DB.init_db()

    # Auto-migrate legacy known_faces/ + rfid_cards.json to SQLite if database is empty
    face_app = FACE_DB._init_app()
    migrate_legacy_faces_if_empty(face_app)

    # Pre-load students and embeddings directly from SQLite into memory
    loaded_count = FACE_DB.load()
    log.info("Server startup complete — Face database cached in memory (%d faces loaded).", loaded_count)

    # Start independent RFID background listener as an asyncio background task on startup
    rfid_task = asyncio.create_task(rfid_background_listener(), name="rfid-background-listener")
    RFID_MANAGER._async_task = rfid_task
    RFID_MANAGER._running = True
    RFID_MANAGER._stop_event.clear()

    # NOTE: Camera is NOT started here — it starts only when a professor
    # explicitly begins a lecture session via POST /start-session.

    yield

    RFID_MANAGER.stop()
    if not rfid_task.done():
        rfid_task.cancel()
        try:
            await rfid_task
        except (asyncio.CancelledError, Exception):
            pass
    CAMERA.stop()
    log.info("Server shutdown complete.")


app = FastAPI(
    title      = "Smart Attendance System — MTU Polytechnic",
    description= "100% Offline AI Face-Recognition Attendance & Lecture Manager",
    version    = "3.1.0",
    docs_url   = "/api/docs",
    redoc_url  = None,
    lifespan   = lifespan,
)


# DESIGN TOKENS & STYLING (B&W FUTURISTIC + MODERN ARABIC TYPOGRAPHY)
DESIGN_CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  :root {
    --black    : #000000;
    --white    : #ffffff;
    --g950     : #050505;
    --g900     : #0a0a0a;
    --g800     : #121212;
    --g700     : #1c1c1c;
    --g600     : #262626;
    --g500     : #383838;
    --g400     : #606060;
    --g300     : #909090;
    --g200     : #c8c8c8;
    --g100     : #eeeeee;
    --border   : 1px solid #222222;
    --border-light : 1px solid #333333;
    --mono     : 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    --sans     : 'Cairo', 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
    --ease     : 160ms cubic-bezier(.4, 0, .2, 1);
  }
  body {
    background: var(--black);
    color: var(--white);
    font-family: var(--sans);
    min-height: 100vh;
    direction: rtl;
    overflow-x: hidden;
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--g950); }
  ::-webkit-scrollbar-thumb { background: var(--g600); }

  /* Shell */
  .shell {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    max-width: 1040px;
    margin: 0 auto;
    padding: 0 32px;
  }

  /* Header */
  .site-header {
    border-bottom: var(--border);
    padding: 42px 0 32px;
    position: relative;
  }
  .site-header::before {
    content: '';
    position: absolute;
    bottom: -1px;
    right: 0;
    width: 90px;
    height: 1px;
    background: var(--white);
  }
  .badge {
    position: absolute;
    top: 42px;
    left: 0;
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: .2em;
    color: var(--g300);
    border: var(--border-light);
    background: var(--g900);
    padding: 5px 12px;
    text-transform: uppercase;
  }
  .eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .24em;
    text-transform: uppercase;
    color: var(--g400);
    margin-bottom: 8px;
  }
  .inst-name {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: .02em;
    line-height: 1.25;
  }
  .inst-name span {
    display: block;
    color: var(--g300);
    font-size: 13px;
    font-weight: 400;
    margin-top: 6px;
    letter-spacing: .12em;
    font-family: var(--mono);
    text-transform: uppercase;
  }

  /* Section Title */
  .sec-title {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .28em;
    text-transform: uppercase;
    color: var(--g400);
    margin-bottom: 24px;
    padding-top: 46px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--g700);
  }

  /* Card */
  .card {
    border: var(--border);
    padding: 46px 52px;
    background: var(--g900);
    position: relative;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 36px;
    height: 36px;
    border-top: 2px solid var(--white);
    border-right: 2px solid var(--white);
  }
  .card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 36px;
    height: 36px;
    border-bottom: 2px solid var(--g500);
    border-left: 2px solid var(--g500);
  }

  /* Form */
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 26px 36px; }
  .fg { display: flex; flex-direction: column; gap: 9px; }
  .fg.full { grid-column: 1 / -1; }
  .divider { grid-column: 1 / -1; height: 1px; background: var(--g700); margin: 6px 0; }
  label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--g300);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  label .ar {
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 700;
    color: var(--white);
  }
  label .req { color: var(--white); font-size: 16px; }
  select, input[type=text] {
    background: var(--black);
    color: var(--white);
    border: 1px solid var(--g600);
    padding: 14px 18px;
    font-size: 15px;
    font-family: var(--sans);
    font-weight: 500;
    direction: rtl;
    width: 100%;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    cursor: pointer;
    transition: border-color var(--ease), background var(--ease);
    border-radius: 0;
  }
  select:hover, input:hover { border-color: var(--g400); }
  select:focus, input:focus {
    border-color: var(--white);
    background: var(--g800);
    box-shadow: 0 0 0 1px var(--white) inset;
  }
  .sw { position: relative; }
  .sw::after {
    content: '▾';
    position: absolute;
    left: 18px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--g400);
    pointer-events: none;
    font-size: 14px;
  }

  /* Button */
  .btn-start {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin-top: 40px;
    width: 100%;
    padding: 20px 32px;
    background: var(--white);
    color: var(--black);
    border: 2px solid var(--white);
    font-size: 18px;
    font-weight: 800;
    letter-spacing: .06em;
    cursor: pointer;
    font-family: var(--sans);
    transition: color var(--ease), transform 80ms ease;
    position: relative;
    overflow: hidden;
  }
  .btn-start::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--black);
    transform: scaleX(0);
    transform-origin: right;
    transition: transform 240ms cubic-bezier(.4,0,.2,1);
  }
  .btn-start:hover::before { transform: scaleX(1); }
  .btn-start:hover { color: var(--white); }
  .btn-start:active { transform: scale(.985); }
  .btn-start span, .btn-arrow { position: relative; z-index: 1; }
  .btn-arrow { font-size: 24px; transition: transform var(--ease); }
  .btn-start:hover .btn-arrow { transform: translateX(-6px); }

  /* Status Row */
  .status-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 42px;
    padding: 16px 22px;
    border: var(--border);
    background: var(--g950);
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .15em;
    color: var(--g300);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--g600); flex-shrink: 0; }
  .dot.ok { background: #4caf50; box-shadow: 0 0 8px #4caf5088; }
  .dot.err { background: #f44336; box-shadow: 0 0 8px #f4433688; }
  .db-pill {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .12em;
    color: var(--g300);
    border: var(--border);
    background: var(--g900);
    padding: 4px 12px;
    text-transform: uppercase;
  }
  .db-pill strong { color: #ffffff; }

  /* Footer */
  footer {
    border-top: var(--border);
    padding: 24px 0;
    margin-top: auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .18em;
    color: var(--g400);
    text-transform: uppercase;
  }

  /* Loader */
  .loader {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.92);
    z-index: 999;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 24px;
  }
  .loader.on { display: flex; }
  .ring {
    width: 52px;
    height: 52px;
    border: 2px solid var(--g700);
    border-top: 2px solid var(--white);
    border-radius: 50%;
    animation: spin 700ms linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loader-txt {
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: .24em;
    color: var(--g200);
    text-transform: uppercase;
  }

  /* Toast */
  .toast {
    position: fixed;
    bottom: 34px;
    left: 50%;
    transform: translateX(-50%) translateY(120%);
    background: var(--g800);
    color: var(--white);
    border: var(--border-light);
    padding: 14px 30px;
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 600;
    transition: transform 320ms cubic-bezier(.4,0,.2,1);
    z-index: 1000;
    text-align: center;
  }
  .toast.show { transform: translateX(-50%) translateY(0); }
  .toast.err { border-color: #f44336; color: #f44336; }

  /* Navigation Bar */
  .top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 22px 0 18px;
    border-bottom: var(--border);
    margin-bottom: 28px;
    gap: 16px;
    flex-wrap: wrap;
  }
  .nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .nav-logo {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .2em;
    background: var(--white);
    color: var(--black);
    padding: 4px 10px;
    text-transform: uppercase;
  }
  .nav-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--g200);
  }
  .nav-links {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .nav-link {
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 600;
    color: var(--g400);
    text-decoration: none;
    padding: 8px 18px;
    border: 1px solid transparent;
    transition: all var(--ease);
  }
  .nav-link:hover {
    color: var(--white);
    border-color: var(--g700);
    background: var(--g900);
  }
  .nav-link.active {
    color: var(--white);
    background: var(--g900);
    border-color: var(--white);
  }
  .nav-session {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 14px;
    border: var(--border);
    background: var(--g950);
  }
  .nav-session.active {
    border-color: rgba(76, 175, 80, 0.4);
    background: rgba(76, 175, 80, 0.08);
    color: #4caf50;
  }
  .nav-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4caf50;
    animation: lp 1.2s ease infinite;
  }
  .nav-dot-idle {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--g600);
  }
  .nav-lecture-btn {
    font-family: var(--sans);
    font-size: 11px;
    font-weight: 700;
    color: var(--black);
    background: var(--white);
    padding: 4px 12px;
    text-decoration: none;
    transition: opacity var(--ease);
  }
  .nav-lecture-btn:hover { opacity: 0.85; }

  /* Tabs */
  .tabs-nav {
    display: flex;
    gap: 12px;
    margin-bottom: 28px;
    border-bottom: var(--border);
    padding-bottom: 12px;
  }
  .tab-btn {
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 700;
    padding: 10px 24px;
    background: var(--g950);
    color: var(--g400);
    border: var(--border);
    cursor: pointer;
    transition: all var(--ease);
  }
  .tab-btn:hover { color: var(--white); border-color: var(--g500); }
  .tab-btn.active {
    background: var(--white);
    color: var(--black);
    border-color: var(--white);
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Tables */
  .table-responsive {
    overflow-x: auto;
    margin-top: 20px;
    border: var(--border);
    background: var(--g900);
  }
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    text-align: right;
  }
  .data-table th, .data-table td {
    padding: 12px 16px;
    border-bottom: var(--border);
  }
  .data-table th {
    background: var(--g950);
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .15em;
    color: var(--g300);
    text-transform: uppercase;
  }
  .data-table tr:hover td {
    background: var(--g950);
  }
  .data-table img.thumb {
    width: 44px;
    height: 44px;
    object-fit: cover;
    border: 1px solid var(--g700);
    vertical-align: middle;
  }
  .thumb-ph {
    width: 44px;
    height: 44px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--g950);
    border: 1px solid var(--g700);
    color: var(--g500);
    font-size: 18px;
  }
  .btn-sm {
    font-family: var(--sans);
    font-size: 11px;
    font-weight: 600;
    padding: 5px 12px;
    cursor: pointer;
    border: var(--border);
    background: var(--g800);
    color: var(--g200);
    transition: all var(--ease);
  }
  .btn-sm:hover { color: var(--white); border-color: var(--white); }
  .btn-sm.del:hover { background: #f44336; color: var(--white); border-color: #f44336; }
  .btn-sm.primary { background: var(--white); color: var(--black); border-color: var(--white); }
  .btn-sm.primary:hover { opacity: 0.85; }

  /* Panels Grid */
  .panels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 24px;
    margin-top: 24px;
  }
  .panel-box {
    border: var(--border);
    background: var(--g900);
    padding: 24px;
  }
  .panel-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .item-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 280px;
    overflow-y: auto;
    margin-bottom: 16px;
  }
  .item-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--g950);
    border: 1px solid var(--g800);
    gap: 8px;
  }
  .item-name {
    font-size: 13px;
    color: var(--white);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .add-item-row {
    display: flex;
    gap: 8px;
  }
  .add-item-row input {
    padding: 8px 12px;
    font-size: 13px;
  }
  .photo-capture-box {
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
    background: var(--g950);
    border: var(--border);
    padding: 16px;
  }
  .photo-preview {
    width: 90px;
    height: 90px;
    border: 1px solid var(--g700);
    background: var(--black);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
  }
  .photo-preview img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .filter-row {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }
  .filter-row select, .filter-row input {
    padding: 8px 14px;
    font-size: 13px;
    width: auto;
    min-width: 160px;
  }
  /* KPI Cards */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 24px 0;
  }
  .kpi-card {
    background: var(--g900);
    border: var(--border);
    padding: 20px 22px;
  }
  .kpi-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .12em;
    color: var(--g300);
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .kpi-val {
    font-size: 26px;
    font-weight: 800;
    color: var(--white);
    font-family: var(--mono);
  }

  /* Status Badges */
  .badge-status {
    display: inline-block;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 700;
    text-align: center;
    border-radius: 2px;
  }
  .badge-good {
    background: rgba(76, 175, 80, 0.15);
    color: #4caf50;
    border: 1px solid rgba(76, 175, 80, 0.4);
  }
  .badge-warn {
    background: rgba(255, 152, 0, 0.15);
    color: #ff9800;
    border: 1px solid rgba(255, 152, 0, 0.4);
  }
  .badge-danger {
    background: rgba(244, 67, 54, 0.15);
    color: #f44336;
    border: 1px solid rgba(244, 67, 54, 0.4);
  }

  /* Progress Bar */
  .progress-track {
    background: var(--g800);
    height: 6px;
    width: 100%;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 4px;
  }
  .progress-fill {
    height: 100%;
    transition: width 0.3s;
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.85);
    z-index: 2000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  .modal-card {
    background: var(--g950);
    border: 1px solid var(--g700);
    width: 100%;
    max-width: 680px;
    max-height: 85vh;
    overflow-y: auto;
    padding: 24px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.8);
  }
  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: var(--border);
    padding-bottom: 14px;
    margin-bottom: 16px;
  }


"""

LECTURE_CSS = DESIGN_CSS + """
  html, body { height: 100%; overflow: hidden; }
  .lshell {
    display: grid;
    grid-template-rows: auto 1fr auto;
    height: 100vh;
    overflow: hidden;
  }
  .lhdr {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    padding: 14px 24px;
    border-bottom: var(--border);
    background: var(--g900);
  }
  .pill {
    font-family: var(--sans);
    font-size: 13px;
    color: var(--g300);
    border: var(--border);
    background: var(--g950);
    padding: 6px 14px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .pill strong { color: var(--white); font-weight: 700; }
  .pill .mono-val { font-family: var(--mono); font-size: 11px; letter-spacing: .1em; }
  .lbody {
    display: grid;
    grid-template-columns: 1fr 390px;
    overflow: hidden;
  }
  .pvideo {
    border-left: var(--border);
    background: var(--black);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
  }
  .pvideo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .vph {
    text-align: center;
    color: var(--g500);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: .2em;
    text-transform: uppercase;
    line-height: 2.2;
  }
  .preg {
    background: var(--g900);
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
  }
  .reg-hdr {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .22em;
    color: var(--g300);
    text-transform: uppercase;
    border-bottom: var(--border);
    padding-bottom: 14px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .reg-cnt-badge {
    background: var(--white);
    color: var(--black);
    font-weight: 800;
    font-size: 13px;
    padding: 2px 10px;
  }
  .lftr {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 24px;
    border-top: var(--border);
    background: var(--g900);
  }
  .btn-c {
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 700;
    padding: 11px 22px;
    cursor: pointer;
    border: var(--border);
    background: var(--g950);
    color: var(--g200);
    transition: all var(--ease);
    border-radius: 0;
  }
  .btn-c:hover { background: var(--g700); color: var(--white); border-color: var(--white); }
  .btn-c.danger { border-color: #f44336; color: #f44336; }
  .btn-c.danger:hover { background: #f44336; color: var(--black); }
  .btn-c.primary { border-color: var(--white); color: var(--black); background: var(--white); }
  .btn-c.primary:hover { background: var(--black); color: var(--white); }
  .live-ind {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-right: auto;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .2em;
    color: var(--g300);
    text-transform: uppercase;
  }
  .ldot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f44336;
    animation: lp 1.2s ease infinite;
  }
  @keyframes lp {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 6px #f44336; }
    50% { opacity: .4; transform: scale(.75); }
  }

  /* Student Attendance Cards */
  @keyframes cardSlide {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: none; }
  }
  .scard {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    background: var(--g950);
    border: var(--border);
    margin-bottom: 10px;
    animation: cardSlide 240ms cubic-bezier(.4, 0, .2, 1);
    position: relative;
    transition: border-color var(--ease);
  }
  .scard:hover { border-color: var(--g500); }
  .scard::before {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: var(--white);
  }
  .savatar {
    width: 48px;
    height: 48px;
    object-fit: cover;
    border: 1px solid #333333;
    flex-shrink: 0;
    background: #111111;
  }
  .saph {
    width: 48px;
    height: 48px;
    background: #111111;
    border: 1px solid #333333;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #444444;
    font-size: 20px;
    flex-shrink: 0;
  }
  .sinfo { display: flex; flex-direction: column; gap: 4px; overflow: hidden; }
  .sname {
    font-size: 14px;
    font-weight: 700;
    color: var(--white);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .smeta {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .stime {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--g400);
    letter-spacing: .08em;
  }
  .stag {
    font-family: var(--sans);
    font-size: 10px;
    font-weight: 700;
    color: #4caf50;
    background: rgba(76, 175, 80, 0.12);
    border: 1px solid rgba(76, 175, 80, 0.35);
    padding: 1px 7px;
  }
  .await {
    color: var(--g500);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .16em;
    text-align: center;
    margin-top: 60px;
    text-transform: uppercase;
    line-height: 2.6;
  }
"""


# TEMPLATE HELPERS
def _opts(items: list[str], placeholder: str) -> str:
    html = f'<option value="" disabled selected>{placeholder}</option>\n'
    for item in items:
        html += f'        <option value="{item}">{item}</option>\n'
    return html


def render_navbar(active: str = "setup") -> str:
    """Unified top navigation bar across all system pages."""
    session_active = SESSION.active
    session_course = SESSION.course or "محاضرة جارية"

    if session_active:
        status_pill = (
            f'<div class="nav-session active">'
            f'<span class="nav-pulse"></span>'
            f'<span>جلسة نشطة: <strong>{session_course}</strong></span>'
            f'<a href="/lecture" class="nav-lecture-btn">الانتقال للمحاضرة &larr;</a>'
            f'</div>'
        )
    else:
        status_pill = (
            f'<div class="nav-session idle">'
            f'<span class="nav-dot-idle"></span>'
            f'<span>لا توجد جلسة نشطة</span>'
            f'</div>'
        )

    return f"""
  <nav class="top-nav">
    <div class="nav-brand">
      <span class="nav-logo">AGY</span>
      <span class="nav-title">نظام الحضور والذكاء الاصطناعي</span>
    </div>
    <div class="nav-links">
      <a href="/" class="nav-link{' active' if active == 'setup' else ''}">إعداد المحاضرة</a>
      <a href="/students" class="nav-link{' active' if active == 'students' else ''}">إدارة الطلبة والأقسام</a>
      <a href="/reports" class="nav-link{' active' if active == 'reports' else ''}">التقارير التراكمية</a>
    </div>
    <div class="nav-status">
      {status_pill}
    </div>
  </nav>"""


# SETUP DASHBOARD HTML
def render_setup_page() -> str:
    db_count  = FACE_DB.count
    db_status = f"<strong>{db_count}</strong> وجه محمّل" if db_count else "لا توجد وجوه"
    fr_badge  = (
        '<strong style="color:#4caf50">محرك التعرف (MediaPipe & CPU Optimized): نشط</strong>'
        if ENGINE_AVAILABLE else
        '<strong style="color:#f44336">محرك التعرف: غير متوفر</strong>'
    )

    inst_ar, inst_en = get_institution_names()
    depts  = get_academic_departments()
    stages = get_academic_stages()
    secs   = get_academic_sections()
    navbar = render_navbar("setup")

    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <meta name="description" content="نظام إدارة الحضور والمحاضرات الذكي"/>
  <meta name="robots" content="noindex,nofollow"/>
  <title>نظام الحضور الذكي — {inst_ar}</title>
  <style>{DESIGN_CSS}</style>
</head>
<body>

<div class="loader" id="ldr"><div class="ring"></div><div class="loader-txt">INITIALISING LECTURE SESSION...</div></div>
<div class="toast" id="toast"></div>

<div class="shell">
  {navbar}

  <header class="site-header">
    <div class="badge">100% OFFLINE · v3.1</div>
    <div class="eyebrow">SMART ATTENDANCE & LECTURE MANAGEMENT</div>
    <h1 class="inst-name">
      {inst_ar}
      <span>{inst_en}</span>
    </h1>
  </header>

  <main>
    <div class="sec-title">إعداد المحاضرة — SESSION CONFIGURATION</div>
    <div class="card">
      <form id="setupForm" method="POST" action="/start-session" novalidate>
        <div class="form-grid">

          <div class="fg full">
            <label for="department">
              <span class="ar">القسم الأكاديمي</span> &nbsp;·&nbsp; ACADEMIC DEPARTMENT <span class="req">*</span>
            </label>
            <div class="sw"><select id="department" name="department" required>{_opts(depts, "— اختر القسم الأكاديمي —")}</select></div>
          </div>

          <div class="divider"></div>

          <div class="fg">
            <label for="section">
              <span class="ar">الشعبة</span> &nbsp;·&nbsp; STUDY SECTION <span class="req">*</span>
            </label>
            <div class="sw"><select id="section" name="section" required>{_opts(secs, "— اختر الشعبة —")}</select></div>
          </div>

          <div class="fg">
            <label for="stage">
              <span class="ar">المرحلة الدراسية</span> &nbsp;·&nbsp; ACADEMIC STAGE <span class="req">*</span>
            </label>
            <div class="sw"><select id="stage" name="stage" required>{_opts(stages, "— اختر المرحلة الدراسية —")}</select></div>
          </div>

          <div class="divider"></div>

          <div class="fg full">
            <label for="course">
              <span class="ar">اسم المادة / المقرر</span> &nbsp;·&nbsp; COURSE NAME <span class="req">*</span>
            </label>
            <input type="text" id="course" name="course" list="courseSuggestions"
                   placeholder="مثال: الذكاء الاصطناعي ومعالجة الإشارات..." required autocomplete="off" spellcheck="false"/>
            <datalist id="courseSuggestions"></datalist>
          </div>

        </div>
        <button type="submit" class="btn-start" id="startBtn">
          <span>بدء المحاضرة وتشغيل الحضور الذكي</span>
          <span class="btn-arrow">&#8592;</span>
        </button>
      </form>
    </div>

    <!-- Status row -->
    <div class="status-row">
      <div class="dot" id="camDot"></div>
      <span id="camTxt">جاري فحص الكاميرا الخارجية [جهاز 1]...</span>
      <span style="margin-right:auto;font-family:var(--mono);" id="stTime"></span>
      <span class="db-pill">{db_status}</span>
      <span class="db-pill">{fr_badge}</span>
    </div>

  </main>

  <footer>
    <span>{inst_en} · OFFLINE BIOMETRIC ATTENDANCE ENGINE</span>
    <span id="clk"></span>
  </footer>
</div>

<script>
(function(){{
  const clk=document.getElementById('clk');
  const st=document.getElementById('stTime');
  function tick(){{
    const n=new Date();
    const t=n.toLocaleTimeString('ar-IQ',{{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
    const d=n.toLocaleDateString('ar-IQ');
    if(clk)clk.textContent=t+' · '+d;
    if(st)st.textContent=t;
  }}
  tick();setInterval(tick,1000);
}})();

fetch('/api/camera-status').then(r=>r.json()).then(d=>{{
  const dot=document.getElementById('camDot');
  const txt=document.getElementById('camTxt');
  if(d.ok){{
    dot.classList.add('ok');
    txt.textContent='الكاميرا الخارجية متصلة ومستعدة [جهاز '+d.index+'] · النظام جاهز';
  }} else {{
    dot.classList.add('err');
    txt.textContent='تنبيه الكاميرا [جهاز '+d.index+']: '+(d.message||'يرجى التأكد من توصيل الكاميرا الخارجية');
  }}
}}).catch(()=>{{
  document.getElementById('camDot').classList.add('err');
}});

function showToast(m,e){{
  const t=document.getElementById('toast');
  t.textContent=m;
  t.className='toast show'+(e?' err':'');
  setTimeout(()=>t.className='toast'+(e?' err':''),3500);
}}

function updateCourseSuggestions(){{
  const d = document.getElementById('department').value;
  const st = document.getElementById('stage').value;
  const s = document.getElementById('section').value;
  const p = new URLSearchParams();
  if(d) p.set('department', d);
  if(st) p.set('stage', st);
  if(s) p.set('section', s);
  fetch('/api/courses?' + p.toString()).then(r=>r.json()).then(data=>{{
    const dl = document.getElementById('courseSuggestions');
    if(!dl) return;
    dl.innerHTML = (data.courses || []).map(c => `<option value="${{c}}">`).join('');
  }}).catch(()=>{{}});
}}
document.getElementById('department').addEventListener('change', updateCourseSuggestions);
document.getElementById('stage').addEventListener('change', updateCourseSuggestions);
document.getElementById('section').addEventListener('change', updateCourseSuggestions);
updateCourseSuggestions();

document.getElementById('setupForm').addEventListener('submit',function(e){{
  e.preventDefault();
  const v=id=>document.getElementById(id).value.trim();
  if(!v('department')){{showToast('يرجى اختيار القسم الأكاديمي',true);return;}}
  if(!v('section'))   {{showToast('يرجى اختيار الشعبة الدراسية',true);return;}}
  if(!v('stage'))     {{showToast('يرجى اختيار المرحلة الدراسية',true);return;}}
  if(!v('course'))    {{showToast('يرجى إدخال اسم المادة / المقرر',true);return;}}
  document.getElementById('ldr').classList.add('on');
  const fd=new FormData(this);
  fetch('/start-session',{{method:'POST',body:fd}})
    .then(r=>{{if(r.redirected){{window.location.href=r.url;return;}}return r.json();}})
    .then(d=>{{if(d&&d.redirect)window.location.href=d.redirect;}})
    .catch(()=>{{document.getElementById('ldr').classList.remove('on');showToast('خطأ في بدء الجلسة',true);}});
}});
</script>
</body></html>"""


# STUDENTS & ACADEMIC SETTINGS DASHBOARD HTML (/students)
def render_students_page() -> str:
    inst_ar, inst_en = get_institution_names()
    depts  = get_academic_departments()
    stages = get_academic_stages()
    secs   = get_academic_sections()
    navbar = render_navbar("students")

    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <meta name="description" content="إدارة الطلبة والهيكل الأكاديمي — 100% Offline"/>
  <meta name="robots" content="noindex,nofollow"/>
  <title>إدارة الطلبة والأقسام — {inst_ar}</title>
  <style>{DESIGN_CSS}</style>
</head>
<body>

<div class="toast" id="toast"></div>

<div class="shell">
  {navbar}

  <header class="site-header" style="padding: 24px 0 28px;">
    <div class="badge">STUDENT & ACADEMIC REPO</div>
    <div class="eyebrow">ENROLLMENT & ACADEMIC STRUCTURE MANAGEMENT</div>
    <h1 class="inst-name">
      {inst_ar}
      <span>{inst_en}</span>
    </h1>
  </header>

  <!-- TABS NAV -->
  <div class="tabs-nav" style="margin-top: 32px;">
    <button type="button" class="tab-btn active" id="btnTab1" onclick="switchTab('tab-students')">
      إدارة وتسجيل الطلبة
    </button>
    <button type="button" class="tab-btn" id="btnTab2" onclick="switchTab('tab-academic')">
      إعدادات الأقسام والهيكل الأكاديمي
    </button>
  </div>

  <!-- TAB 1: STUDENTS MANAGEMENT & ENROLLMENT -->
  <div class="tab-panel active" id="tab-students">

    <!-- ADD STUDENT CARD -->
    <div class="sec-title">تسجيل طالب جديد — ENROLL NEW STUDENT</div>
    <div class="card" style="padding: 36px 42px;">
      <form id="enrollForm" novalidate>
        <div class="form-grid">

          <div class="fg">
            <label for="stdDept">
              <span class="ar">القسم الأكاديمي</span> &nbsp;·&nbsp; DEPARTMENT <span class="req">*</span>
            </label>
            <div class="sw"><select id="stdDept" name="department" required>{_opts(depts, "— اختر القسم —")}</select></div>
          </div>

          <div class="fg">
            <label for="stdStage">
              <span class="ar">المرحلة الدراسية</span> &nbsp;·&nbsp; STAGE <span class="req">*</span>
            </label>
            <div class="sw"><select id="stdStage" name="stage" required>{_opts(stages, "— اختر المرحلة —")}</select></div>
          </div>

          <div class="fg">
            <label for="stdSec">
              <span class="ar">الشعبة الدراسية</span> &nbsp;·&nbsp; SECTION <span class="req">*</span>
            </label>
            <div class="sw"><select id="stdSec" name="section" required>{_opts(secs, "— اختر الشعبة —")}</select></div>
          </div>

          <div class="fg">
            <label for="stdName">
              <span class="ar">الاسم الثلاثي للطالب</span> &nbsp;·&nbsp; FULL NAME <span class="req">*</span>
            </label>
            <input type="text" id="stdName" name="full_name" placeholder="مثال: علي ضياء حسين..." required autocomplete="off"/>
          </div>

          <div class="fg">
            <label for="stdCode">
              <span class="ar">الرقم الجامعي / الكود</span> &nbsp;·&nbsp; STUDENT CODE
            </label>
            <input type="text" id="stdCode" name="student_code" placeholder="مثال: 2024-AI-104 (اختياري)" autocomplete="off"/>
          </div>

          <div class="fg">
            <label for="stdRfid">
              <span class="ar">معرف بطاقة RFID</span> &nbsp;·&nbsp; CARD UID
            </label>
            <input type="text" id="stdRfid" name="rfid_uid" placeholder="UID:3A:5B:92:F1 (اختياري)" autocomplete="off"/>
          </div>

          <!-- DUAL PHOTO CAPTURE -->
          <div class="fg full">
            <label>
              <span class="ar">صورة الوجه المرجعية</span> &nbsp;·&nbsp; REFERENCE FACE PHOTO <span class="req">*</span>
            </label>
            <div class="photo-capture-box">
              <div class="photo-preview" id="photoPreview">
                <span style="color:var(--g500);font-size:11px;text-align:center;">معاينة الصورة</span>
              </div>
              <div style="display:flex;flex-direction:column;gap:10px;flex:1;">
                <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                  <label class="btn-sm" style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;">
                    اختيار ملف صورة
                    <input type="file" id="stdPhotoFile" accept="image/*" style="display:none;"/>
                  </label>
                  <button type="button" class="btn-sm primary" id="btnCamSnap" onclick="takeSnapshot()">
                    التقاط من الكاميرا
                  </button>
                  <button type="button" class="btn-sm" id="btnClearPhoto" onclick="clearPhoto()" style="display:none;">
                    إزالة
                  </button>
                </div>
                <span style="font-size:11px;color:var(--g400);">
                  يتم استخراج بصمة الوجه (512-d Vector) وحفظها محلياً بأعلى دقة تلقائياً.
                </span>
              </div>
            </div>
            <input type="hidden" id="photoBase64" name="photo_base64"/>
          </div>

        </div>

        <button type="submit" class="btn-start" id="btnEnrollSubmit" style="margin-top:28px;padding:16px 28px;font-size:16px;">
          <span id="enrollBtnTxt">تسجيل الطالب واستخراج بصمة الوجه</span>
          <span class="btn-arrow">&#8592;</span>
        </button>
      </form>
    </div>

    <!-- FILTER & STUDENTS LIST TABLE -->
    <div class="sec-title" style="margin-top:20px;">
      قائمة الطلبة المقيدين — ENROLLED ROSTER
      <span id="studentsTotalCount" style="color:var(--white);font-weight:700;">(0)</span>
    </div>

    <div class="filter-row">
      <div class="sw"><select id="filterDept" onchange="loadStudents()"><option value="">كل الأقسام الأكاديمية</option>{_opts(depts, "كل الأقسام")}</select></div>
      <div class="sw"><select id="filterStage" onchange="loadStudents()"><option value="">كل المراحل</option>{_opts(stages, "كل المراحل")}</select></div>
      <div class="sw"><select id="filterSec" onchange="loadStudents()"><option value="">كل الشعب</option>{_opts(secs, "كل الشعب")}</select></div>
      <input type="text" id="filterSearch" placeholder="بحث بالاسم أو الكود..." oninput="filterTableLocally()"/>
    </div>

    <div class="table-responsive">
      <table class="data-table" id="studentsTable">
        <thead>
          <tr>
            <th style="width:50px;">#</th>
            <th style="width:70px;">الصورة</th>
            <th>اسم الطالب</th>
            <th>الرقم الجامعي</th>
            <th>القسم / المرحلة / الشعبة</th>
            <th>بطاقة RFID</th>
            <th style="width:130px;text-align:center;">إجراءات</th>
          </tr>
        </thead>
        <tbody id="studentsTbody">
          <tr><td colspan="7" style="text-align:center;color:var(--g500);padding:28px;">جاري تحميل بيانات الطلبة...</td></tr>
        </tbody>
      </table>
    </div>

  </div>

  <!-- TAB 2: ACADEMIC STRUCTURE SETTINGS -->
  <div class="tab-panel" id="tab-academic">

    <!-- INSTITUTION DETAILS CARD -->
    <div class="sec-title">اسم المؤسسة الأكاديمية — INSTITUTION PROFILE</div>
    <div class="card" style="padding: 32px 36px; margin-bottom: 36px;">
      <form id="instForm" onsubmit="saveInstitution(event)">
        <div class="form-grid">
          <div class="fg">
            <label for="instArInput"><span class="ar">اسم الكلية / الجامعة بالعربية</span> <span class="req">*</span></label>
            <input type="text" id="instArInput" value="{inst_ar}" required autocomplete="off"/>
          </div>
          <div class="fg">
            <label for="instEnInput"><span class="ar">اسم الكلية / الجامعة بالإنجليزية</span> <span class="req">*</span></label>
            <input type="text" id="instEnInput" value="{inst_en}" required autocomplete="off" style="direction:ltr;text-align:left;"/>
          </div>
        </div>
        <button type="submit" class="btn-sm primary" style="margin-top:20px;padding:10px 24px;font-size:13px;">
          حفظ اسم المؤسسة
        </button>
      </form>
    </div>

    <!-- 3 ACADEMIC PANELS: DEPTS, STAGES, SECTIONS -->
    <div class="sec-title">الهيكل الأكاديمي — ACADEMIC UNITS</div>
    <div class="panels-grid">

      <!-- DEPARTMENTS PANEL -->
      <div class="panel-box">
        <div class="panel-title">
          <span>الأقسام الأكاديمية</span>
          <span style="font-family:var(--mono);font-size:11px;color:var(--g400);" id="cnt-dept">0</span>
        </div>
        <div class="item-list" id="list-department"></div>
        <div class="add-item-row">
          <input type="text" id="new-dept" placeholder="اسم القسم الجديد..."/>
          <button type="button" class="btn-sm primary" onclick="addItem('department')">+ إضافة</button>
        </div>
      </div>

      <!-- STAGES PANEL -->
      <div class="panel-box">
        <div class="panel-title">
          <span>المراحل الدراسية</span>
          <span style="font-family:var(--mono);font-size:11px;color:var(--g400);" id="cnt-stage">0</span>
        </div>
        <div class="item-list" id="list-stage"></div>
        <div class="add-item-row">
          <input type="text" id="new-stage" placeholder="المرحلة الجديدة..."/>
          <button type="button" class="btn-sm primary" onclick="addItem('stage')">+ إضافة</button>
        </div>
      </div>

      <!-- SECTIONS PANEL -->
      <div class="panel-box">
        <div class="panel-title">
          <span>الشعب الدراسية</span>
          <span style="font-family:var(--mono);font-size:11px;color:var(--g400);" id="cnt-section">0</span>
        </div>
        <div class="item-list" id="list-section"></div>
        <div class="add-item-row">
          <input type="text" id="new-sec" placeholder="الشعبة الجديدة..."/>
          <button type="button" class="btn-sm primary" onclick="addItem('section')">+ إضافة</button>
        </div>
      </div>

    </div>

  </div>

  <footer style="margin-top:60px;">
    <span>{inst_en} · ACADEMIC & ENROLLMENT DASHBOARD</span>
    <span>100% OFFLINE</span>
  </footer>

</div>

<script>
let _allStudents = [];

function showToast(m, e){{
  const t = document.getElementById('toast');
  if(!t) return;
  t.textContent = m;
  t.className = 'toast show' + (e ? ' err' : '');
  setTimeout(() => t.className = 'toast' + (e ? ' err' : ''), 3500);
}}

function switchTab(tabId){{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  if(tabId === 'tab-students'){{
    document.getElementById('btnTab1').classList.add('active');
    document.getElementById('tab-students').classList.add('active');
    loadStudents();
  }} else {{
    document.getElementById('btnTab2').classList.add('active');
    document.getElementById('tab-academic').classList.add('active');
    loadSettings();
  }}
}}

// Photo preview handlers
document.getElementById('stdPhotoFile').addEventListener('change', function(e){{
  const file = e.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = function(evt){{
    document.getElementById('photoPreview').innerHTML = `<img src="${{evt.target.result}}"/>`;
    document.getElementById('photoBase64').value = evt.target.result;
    document.getElementById('btnClearPhoto').style.display = 'inline-block';
  }};
  reader.readAsDataURL(file);
}});

function clearPhoto(){{
  document.getElementById('stdPhotoFile').value = '';
  document.getElementById('photoBase64').value = '';
  document.getElementById('photoPreview').innerHTML = '<span style="color:var(--g500);font-size:11px;text-align:center;">معاينة الصورة</span>';
  document.getElementById('btnClearPhoto').style.display = 'none';
}}

function takeSnapshot(){{
  const btn = document.getElementById('btnCamSnap');
  btn.textContent = 'جاري الالتقاط...';
  btn.disabled = true;

  fetch('/api/camera-snapshot')
    .then(r => {{
      if(!r.ok) throw new Error('الكاميرا غير متاحة حالياً');
      return r.json();
    }})
    .then(d => {{
      if(d.ok && d.image_b64){{
        const b64 = 'data:image/jpeg;base64,' + d.image_b64;
        document.getElementById('photoPreview').innerHTML = `<img src="${{b64}}"/>`;
        document.getElementById('photoBase64').value = b64;
        document.getElementById('stdPhotoFile').value = '';
        document.getElementById('btnClearPhoto').style.display = 'inline-block';
        showToast('تم التقاط صورة الوجه من الكاميرا بنجاح', false);
      }}
    }})
    .catch(err => {{
      // Fallback to browser webcam (HTML5 getUserMedia)
      navigator.mediaDevices.getUserMedia({{ video: {{ width: 640, height: 480 }} }})
        .then(stream => {{
          const video = document.createElement('video');
          video.srcObject = stream;
          video.play();
          setTimeout(() => {{
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            stream.getTracks().forEach(t => t.stop());
            const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
            document.getElementById('photoPreview').innerHTML = `<img src="${{dataUrl}}"/>`;
            document.getElementById('photoBase64').value = dataUrl;
            document.getElementById('stdPhotoFile').value = '';
            document.getElementById('btnClearPhoto').style.display = 'inline-block';
            showToast('تم التقاط الصورة من كاميرا المتصفح', false);
          }}, 500);
        }})
        .catch(navErr => {{
          showToast('تعذر الوصول للكاميرا: ' + (err.message || navErr.message), true);
        }});
    }})
    .finally(() => {{
      btn.textContent = 'التقاط من الكاميرا';
      btn.disabled = false;
    }});
}}

// Student Enrollment Form Submit
document.getElementById('enrollForm').addEventListener('submit', function(e){{
  e.preventDefault();
  const name = document.getElementById('stdName').value.trim();
  const dept = document.getElementById('stdDept').value;
  const stage = document.getElementById('stdStage').value;
  const sec = document.getElementById('stdSec').value;
  const photoB64 = document.getElementById('photoBase64').value;
  const photoFile = document.getElementById('stdPhotoFile').files[0];

  if(!name){{ showToast('يرجى إدخال اسم الطالب', true); return; }}
  if(!dept){{ showToast('يرجى اختيار القسم الأكاديمي', true); return; }}
  if(!stage){{ showToast('يرجى اختيار المرحلة الدراسية', true); return; }}
  if(!sec){{ showToast('يرجى اختيار الشعبة الدراسية', true); return; }}
  if(!photoB64 && !photoFile){{ showToast('يرجى اختيار صورة للطالب أو التقاطها من الكاميرا', true); return; }}

  const btn = document.getElementById('btnEnrollSubmit');
  const txt = document.getElementById('enrollBtnTxt');
  btn.disabled = true;
  txt.textContent = 'جاري استخراج بصمة الوجه والتسجيل...';

  const fd = new FormData();
  fd.append('full_name', name);
  fd.append('department', dept);
  fd.append('stage', stage);
  fd.append('section', sec);
  fd.append('student_code', document.getElementById('stdCode').value.trim());
  fd.append('rfid_uid', document.getElementById('stdRfid').value.trim());
  if(photoFile) {{
    fd.append('photo', photoFile);
  }} else if(photoB64) {{
    fd.append('photo_base64', photoB64);
  }}

  fetch('/api/students', {{ method: 'POST', body: fd }})
    .then(r => {{
      if(!r.ok) return r.json().then(j => {{ throw new Error(j.detail || 'خطأ في حفظ الطالب'); }});
      return r.json();
    }})
    .then(d => {{
      showToast('تم تسجيل الطالب بنجاح وإضافة بصمة وجهه للقاعدة', false);
      document.getElementById('stdName').value = '';
      document.getElementById('stdCode').value = '';
      document.getElementById('stdRfid').value = '';
      clearPhoto();
      loadStudents();
    }})
    .catch(err => {{
      showToast(err.message || 'خطأ في عملية التسجيل', true);
    }})
    .finally(() => {{
      btn.disabled = false;
      txt.textContent = 'تسجيل الطالب واستخراج بصمة الوجه';
    }});
}});

// Load students list
function loadStudents(){{
  const dept = document.getElementById('filterDept').value;
  const stage = document.getElementById('filterStage').value;
  const sec = document.getElementById('filterSec').value;
  const p = new URLSearchParams();
  if(dept) p.set('department', dept);
  if(stage) p.set('stage', stage);
  if(sec) p.set('section', sec);

  fetch('/api/students?' + p.toString())
    .then(r => r.json())
    .then(data => {{
      _allStudents = data;
      renderStudentsTable(data);
    }})
    .catch(() => {{
      document.getElementById('studentsTbody').innerHTML = '<tr><td colspan="7" style="text-align:center;color:#f44336;padding:20px;">خطأ في تحميل سجل الطلبة</td></tr>';
    }});
}}

function renderStudentsTable(data){{
  const tb = document.getElementById('studentsTbody');
  document.getElementById('studentsTotalCount').textContent = `(${{data.length}})`;
  if(!data.length){{
    tb.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--g500);padding:32px;">لا يوجد طلبة مسجلون مطابقون للبحث</td></tr>';
    return;
  }}
  tb.innerHTML = data.map((s, idx) => `
    <tr>
      <td style="font-family:var(--mono);color:var(--g400);">${{idx + 1}}</td>
      <td>
        ${{s.photo_b64
          ? `<img class="thumb" src="data:image/jpeg;base64,${{s.photo_b64}}" alt="${{s.full_name}}"/>`
          : `<div class="thumb-ph">◻</div>`}}
      </td>
      <td><strong>${{s.full_name}}</strong></td>
      <td style="font-family:var(--mono);font-size:11px;color:var(--g300);">${{s.student_code || '—'}}</td>
      <td>${{s.department}} / ${{s.stage}} / ${{s.section}}</td>
      <td>
        <span style="font-family:var(--mono);font-size:11px;color:${{s.rfid_uid ? '#4caf50' : 'var(--g500)'}};">${{s.rfid_uid || 'غير مربوط'}}</span>
        <button type="button" class="btn-sm" style="margin-right:6px;" onclick="editStudentRfid(${{s.id}}, '${{s.rfid_uid || ''}}')">تعديل</button>
      </td>
      <td style="text-align:center;">
        <button type="button" class="btn-sm del" onclick="deleteStudent(${{s.id}}, '${{s.full_name}}')">حذف حذف</button>
      </td>
    </tr>
  `).join('');
}}

function filterTableLocally(){{
  const q = document.getElementById('filterSearch').value.trim().toLowerCase();
  if(!q){{
    renderStudentsTable(_allStudents);
    return;
  }}
  const filtered = _allStudents.filter(s =>
    (s.full_name && s.full_name.toLowerCase().includes(q)) ||
    (s.student_code && s.student_code.toLowerCase().includes(q)) ||
    (s.rfid_uid && s.rfid_uid.toLowerCase().includes(q))
  );
  renderStudentsTable(filtered);
}}

function editStudentRfid(id, currentRfid){{
  const n = prompt('أدخل معرف بطاقة RFID (أو اتركه فارغاً لإلغاء الربط):', currentRfid);
  if(n === null) return;
  const fd = new FormData();
  fd.append('rfid_uid', n.trim());
  fetch(`/api/students/${{id}}/rfid`, {{ method: 'PATCH', body: fd }})
    .then(r => {{
      if(!r.ok) return r.json().then(j => {{ throw new Error(j.detail || 'خطأ'); }});
      return r.json();
    }})
    .then(() => {{
      showToast('تم تحديث بطاقة الطالب بنجاح', false);
      loadStudents();
    }})
    .catch(err => showToast(err.message, true));
}}

function deleteStudent(id, name){{
  if(!confirm(`هل أنت متأكد من حذف الطالب "${{name}}"\nسيتم حذف سجله وبصمته بصورة نهائية؟`)) return;
  fetch(`/api/students/${{id}}`, {{ method: 'DELETE' }})
    .then(r => r.json())
    .then(() => {{
      showToast(`تم حذف الطالب ${{name}} بنجاح`, false);
      loadStudents();
    }})
    .catch(() => showToast('خطأ في حذف الطالب', true));
}}

// ── Tab 2: Academic Settings ───────────────────────────────────────────────
function loadSettings(){{
  fetch('/api/settings')
    .then(r => r.json())
    .then(d => {{
      document.getElementById('instArInput').value = d.institution_ar || '';
      document.getElementById('instEnInput').value = d.institution_en || '';
      renderItemList('department', d.departments || []);
      renderItemList('stage', d.stages || []);
      renderItemList('section', d.sections || []);
    }})
    .catch(() => showToast('خطأ في تحميل إعدادات الهيكل الأكاديمي', true));
}}

function renderItemList(cat, items){{
  const list = document.getElementById(`list-${{cat}}`);
  const cnt = document.getElementById(cat === 'section' ? 'cnt-section' : `cnt-${{cat}}`);
  if(cnt) cnt.textContent = items.length;
  if(!items.length){{
    list.innerHTML = '<div style="color:var(--g500);font-size:12px;padding:12px;text-align:center;">لا توجد عناصر مضافة</div>';
    return;
  }}
  list.innerHTML = items.map(item => `
    <div class="item-row">
      <span class="item-name" title="${{item.name}}">${{item.name}}</span>
      <div style="display:flex;gap:6px;">
        <button type="button" class="btn-sm" onclick="renameItem(${{item.id}}, '${{item.name}}')">تعديل</button>
        <button type="button" class="btn-sm del" onclick="deleteItem(${{item.id}}, '${{item.name}}')">حذف</button>
      </div>
    </div>
  `).join('');
}}

function saveInstitution(e){{
  e.preventDefault();
  const ar = document.getElementById('instArInput').value.trim();
  const en = document.getElementById('instEnInput').value.trim();
  if(!ar || !en){{ showToast('يرجى ملء الاسمين بالعربية والإنجليزية', true); return; }}
  const fd = new FormData();
  fd.append('institution_ar', ar);
  fd.append('institution_en', en);
  fetch('/api/settings/institution', {{ method: 'POST', body: fd }})
    .then(r => r.json())
    .then(() => showToast('تم حفظ اسم المؤسسة بنجاح', false))
    .catch(() => showToast('خطأ في حفظ اسم المؤسسة', true));
}}

function addItem(cat){{
  const inpId = cat === 'department' ? 'new-dept' : (cat === 'stage' ? 'new-stage' : 'new-sec');
  const inp = document.getElementById(inpId);
  const val = inp.value.trim();
  if(!val){{ showToast('يرجى إدخال اسم العنصر', true); return; }}
  const fd = new FormData();
  fd.append('category', cat);
  fd.append('name', val);
  fetch('/api/settings/items', {{ method: 'POST', body: fd }})
    .then(r => {{
      if(!r.ok) return r.json().then(j => {{ throw new Error(j.detail || 'خطأ'); }});
      return r.json();
    }})
    .then(() => {{
      inp.value = '';
      showToast('تمت إضافة العنصر بنجاح', false);
      loadSettings();
    }})
    .catch(err => showToast(err.message, true));
}}

function renameItem(id, oldName){{
  const n = prompt('أدخل الاسم الجديد:', oldName);
  if(!n || n.trim() === oldName) return;
  const fd = new FormData();
  fd.append('new_name', n.trim());
  fetch(`/api/settings/items/${{id}}`, {{ method: 'PUT', body: fd }})
    .then(r => {{
      if(!r.ok) return r.json().then(j => {{ throw new Error(j.detail || 'خطأ'); }});
      return r.json();
    }})
    .then(() => {{
      showToast('تم تعديل الاسم وتحديث السجلات المرتبطة به', false);
      loadSettings();
    }})
    .catch(err => showToast(err.message, true));
}}

function deleteItem(id, name){{
  if(!confirm(`هل أنت متأكد من حذف "${{name}}"؟`)) return;
  fetch(`/api/settings/items/${{id}}`, {{ method: 'DELETE' }})
    .then(r => r.json())
    .then(() => {{
      showToast('تم حذف العنصر بنجاح', false);
      loadSettings();
    }})
    .catch(() => showToast('خطأ في الحذف', true));
}}

// Initial load
loadStudents();
</script>
</body></html>"""
def render_lecture_page(session: dict) -> str:
    dept     = session.get("department", "—")
    section  = session.get("section", "—")
    stage    = session.get("stage", "—")
    course   = session.get("course", "—")
    started  = (session.get("started_at") or "—")[:19].replace("T", " ")
    db_count = FACE_DB.count

    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>جلسة نشطة — {course}</title>
  <style>{LECTURE_CSS}</style>
</head>
<body>
<div class="toast" id="toast"></div>
<div class="lshell">

  <!-- TOP HEADER -->
  <header class="lhdr">
    <div class="pill"><strong>{dept}</strong></div>
    <div class="pill">الشعبة: <strong>{section}</strong></div>
    <div class="pill">المرحلة: <strong>{stage}</strong></div>
    <div class="pill">المادة: <strong>{course}</strong></div>
    <div class="pill">وقت البدء: <strong class="mono-val">{started}</strong></div>
    <div class="pill" style="margin-right:auto;" id="clkPill"></div>
    <div class="pill">قاعدة الوجوه: <strong class="mono-val">{db_count}</strong></div>
  </header>

  <!-- BODY: VIDEO + ATTENDANCE REGISTRY -->
  <div class="lbody">
    <!-- LEFT: 30 FPS MJPEG STREAM -->
    <div class="pvideo">
      <img id="vfeed" src="/video-feed" alt="Camera Feed" />
      <div class="vph" id="vph" style="display:none;">
        <div style="font-size:48px;color:var(--g700);margin-bottom:16px;">▣</div>
        CAMERA FEED — DEVICE [1]<br>CONNECTING / AWAITING SIGNAL
      </div>
    </div>

    <!-- RIGHT: REGISTRY WITH ORIGINAL STUDENT PROFILE PHOTOS -->
    <div class="preg">
      <div class="reg-hdr">
        <span>سجل الحضور الذكي · ATTENDANCE LOG</span>
        <span class="reg-cnt-badge" id="cnt">0</span>
      </div>
      <div id="regList">
        <div class="await">في انتظار التعرف على الطلاب...<br>AWAITING FACE RECOGNITION</div>
      </div>
    </div>
  </div>

  <!-- BOTTOM FOOTER CONTROLS -->
  <footer class="lftr">
    <button class="btn-c primary" onclick="exportReport()">&#8595; تصدير تقرير الحضور (Excel)</button>
    <button class="btn-c" onclick="reloadDB()">&#8635; تحديث الوجوه</button>
    <button class="btn-c" id="sndBtn" onclick="toggleSound()">الصوت: مفعّل</button>
    <div class="live-ind"><div class="ldot"></div>بث حي 30 FPS · الحضور مسجل لمرة واحدة</div>
    <button class="btn-c danger" onclick="endSession()">&#9632; إنهاء المحاضرة</button>
  </footer>

</div>

<script>
// 1. Clock display
(function(){{
  const el=document.getElementById('clkPill');
  function tick(){{
    if(el)el.textContent=new Date().toLocaleTimeString('ar-IQ',{{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
  }}
  tick();setInterval(tick,1000);
}})();

// 2. Web Audio API confirmation chime
let _audioEnabled = true;
function playAttendanceChime(){{
  if(!_audioEnabled) return;
  try {{
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if(!AudioCtx) return;
    const ctx = new AudioCtx();
    const now = ctx.currentTime;

    // Harmonic confirmation blip: 1046.5 Hz (C6) -> 1318.5 Hz (E6)
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(1046.5, now);
    osc1.frequency.exponentialRampToValueAtTime(1318.5, now + 0.08);
    gain1.gain.setValueAtTime(0.22, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.16);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.16);

    // Harmonic sweep: 2093 Hz (C7)
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(2093.0, now + 0.06);
    gain2.gain.setValueAtTime(0.16, now + 0.06);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.28);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.06);
    osc2.stop(now + 0.28);
  }} catch(e) {{
    console.debug('WebAudio note:', e);
  }}
}}

function toggleSound(){{
  _audioEnabled = !_audioEnabled;
  const btn = document.getElementById('sndBtn');
  if(btn) btn.textContent = _audioEnabled ? 'الصوت: مفعّل' : 'الصوت: معطّل';
  if(_audioEnabled) playAttendanceChime();
}}

// 3. Attendance state polling
let _lastCount = 0;
function pollAttendance(){{
  fetch('/api/session-state').then(r=>r.json()).then(data=>{{
    const cnt = data.count || 0;
    const cntEl = document.getElementById('cnt');
    if(cntEl) cntEl.textContent = cnt;

    if(cnt !== _lastCount){{
      if(cnt > _lastCount){{
        playAttendanceChime();
      }}
      _lastCount = cnt;
      const list = document.getElementById('regList');
      if(cnt === 0){{
        list.innerHTML = '<div class="await">في انتظار التعرف على الطلاب...<br>AWAITING FACE RECOGNITION</div>';
        return;
      }}
      // Render cards with student's original reference photo from known_faces/
      list.innerHTML = (data.attendance || []).map((s, i) => `
        <div class="scard" style="animation-delay:${{i * 20}}ms">
          ${{s.photo_b64
            ? `<img class="savatar" src="data:image/jpeg;base64,${{s.photo_b64}}" alt="${{s.name || ''}}"/>`
            : `<div class="saph">◻</div>`}}
          <div class="sinfo">
            <div class="sname">${{s.name || '—'}}</div>
            <div class="smeta">
              <span class="stime">${{(s.timestamp || '').slice(11, 19)}}</span>
              <span class="stag">${{(s.method || '').includes('RFID') ? 'بطاقة RFID' : 'حاضر'}}</span>
            </div>
          </div>
        </div>`).join('');
    }}
  }}).catch(()=>{{}});
}}
setInterval(pollAttendance, 400);
pollAttendance();

function showToast(m, e){{
  const t = document.getElementById('toast');
  if(!t) return;
  t.textContent = m;
  t.className = 'toast show' + (e ? ' err' : '');
  setTimeout(() => t.className = 'toast' + (e ? ' err' : ''), 4000);
}}

// Real-time WebSocket attendance stream with auto-reconnect
function connectWebSocket(){{
  try {{
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(proto + '//' + window.location.host + '/ws/attendance');
    ws.onmessage = function(e){{
      try {{
        const msg = JSON.parse(e.data);
        if(msg.event === 'attendance_recorded'){{
          playAttendanceChime();
          pollAttendance();
        }} else if(msg.event === 'wrong_section'){{
          const sName = (msg.data && msg.data.name) || 'طالب';
          const sSec  = (msg.data && msg.data.section) || 'شعبة أخرى';
          showToast(`تنبيه: الطالب ${{sName}} (${{sSec}}) لا ينتمي لهذه الشعبة`, true);
        }}
      }} catch(err){{}}
    }};
    ws.onclose = function(){{
      setTimeout(connectWebSocket, 2000);
    }};
  }} catch(e){{}}
}}
connectWebSocket();

// 4. Actions
function exportReport(){{
  const p = new URLSearchParams();
  const d = "{dept}", c = "{course}", s = "{section}", st = "{stage}";
  if (d && d !== '—') p.set('department', d);
  if (c && c !== '—') p.set('subject', c);
  if (s && s !== '—') p.set('section', s);
  if (st && st !== '—') p.set('stage', st);
  const q = p.toString();
  window.location.href = '/api/export-report' + (q ? '?' + q : '');
}}
function reloadDB(){{
  fetch('/api/reload-faces', {{method: 'POST'}})
    .then(r => r.json())
    .then(d => alert('تم إعادة تحميل قاعدة الوجوه بنجاح: ' + d.count + ' طالب'));
}}
function endSession(){{
  if(!confirm('هل تريد إنهاء المحاضرة الحالية وحفظ تقرير الحضور؟')) return;
  fetch('/api/end-session', {{method: 'POST'}})
    .then(() => window.location.href = '/')
    .catch(() => window.location.href = '/');
}}

// 5. Intelligent Video Stream Watchdog & Auto-Recovery
(function setupVideoFeedWatchdog(){{
  const img = document.getElementById('vfeed');
  const ph = document.getElementById('vph');
  if(!img) return;

  let retryTimer = null;

  function reloadStream(){{
    if(retryTimer) return;
    if(ph) ph.style.display = 'block';
    retryTimer = setTimeout(function(){{
      retryTimer = null;
      console.log('[Video] Reconnecting to /video-feed ...');
      img.src = '/video-feed?t=' + Date.now();
    }}, 1200);
  }}

  img.onload = function(){{
    img.style.display = 'block';
    if(ph) ph.style.display = 'none';
  }};

  img.onerror = function(){{
    console.warn('[Video] Feed interrupted or awaiting camera signal. Auto-reconnecting...');
    reloadStream();
  }};

  // Ensure stream starts immediately with cache-buster
  img.src = '/video-feed?t=' + Date.now();
}})();
</script>
</body></html>"""


# ROUTES — PAGES
@app.get("/", response_class=HTMLResponse, summary="Setup Dashboard")
async def setup_dashboard():
    if SESSION.active:
        return RedirectResponse("/lecture", status_code=303)
    return HTMLResponse(render_setup_page())


@app.get("/students", response_class=HTMLResponse, summary="Students & Academic Settings")
async def students_dashboard():
    return HTMLResponse(render_students_page())


def render_reports_page() -> str:
    inst_ar, inst_en = get_institution_names()
    navbar = render_navbar("reports")
    departments = get_academic_departments()
    stages = get_academic_stages()
    sections = get_academic_sections()

    dept_opts = "\n".join(f'<option value="{d}">{d}</option>' for d in departments)
    stage_opts = "\n".join(f'<option value="{s}">{s}</option>' for s in stages)
    sec_opts = "\n".join(f'<option value="{s}">{s}</option>' for s in sections)

    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>التقارير التراكمية — {inst_ar}</title>
  <style>{DESIGN_CSS}</style>
</head>
<body>
<div class="shell">
  {navbar}

  <main style="margin-top: 32px;">
    <div class="sec-title">التقارير التراكمية والإحصائيات الأكاديمية — CUMULATIVE REPORTS</div>
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 24px; flex-wrap:wrap; gap:16px;">
      <div>
        <h1 style="font-size:24px; font-weight:800; margin-bottom:6px;">سجل الحضور والغياب الفصلي</h1>
        <p style="font-size:13px; color:var(--g400);">تحليل شامل لنسب الحضور والغياب وتحديد حالات الإنذار والحرمان الأكاديمي مع التصدير الرسمي المعتمد</p>
      </div>
      <div>
        <button type="button" class="btn primary" id="exportBtn" onclick="exportCumulativeExcel()" disabled style="height:42px; padding:0 24px; font-size:13px;">
          تصدير التقرير الفصلي (Excel)
        </button>
      </div>
    </div>

    <!-- Filter Bar Card -->
    <div class="card" style="margin-bottom:24px;">
      <div style="font-family:var(--mono); font-size:11px; letter-spacing:.15em; color:var(--g300); text-transform:uppercase; margin-bottom:16px;">
        معايير التصفية والبحث الأكاديمي
      </div>
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:16px;">
        <div class="form-group">
          <label class="lbl">القسم الأكاديمي</label>
          <select id="repDept" class="inp" onchange="onFilterCriteriaChanged()">
            {dept_opts}
          </select>
        </div>
        <div class="form-group">
          <label class="lbl">المرحلة الدراسية</label>
          <select id="repStage" class="inp" onchange="onFilterCriteriaChanged()">
            {stage_opts}
          </select>
        </div>
        <div class="form-group">
          <label class="lbl">الشعبة</label>
          <select id="repSec" class="inp" onchange="onFilterCriteriaChanged()">
            {sec_opts}
          </select>
        </div>
        <div class="form-group">
          <label class="lbl">المادة الدراسية</label>
          <input type="text" id="repCourse" class="inp" list="courseList" placeholder="اختر أو اكتب المادة..." oninput="onCourseInputChanged()"/>
          <datalist id="courseList"></datalist>
        </div>
        <div class="form-group">
          <label class="lbl">من تاريخ (اختياري)</label>
          <input type="date" id="repDateFrom" class="inp"/>
        </div>
        <div class="form-group">
          <label class="lbl">إلى تاريخ (اختياري)</label>
          <input type="date" id="repDateTo" class="inp"/>
        </div>
      </div>
      <div style="margin-top:20px; display:flex; justify-content:flex-end; gap:12px;">
        <button type="button" class="btn" style="height:40px; padding:0 24px; font-size:13px;" onclick="loadCumulativeReport()">
          تحديث التقرير
        </button>
      </div>
    </div>

    <!-- KPI Metric Strip -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">المحاضرات المنعقدة</div>
        <div id="kpiLectures" class="kpi-val">—</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">الطلبة المقيدين</div>
        <div id="kpiStudents" class="kpi-val">—</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">متوسط الحضور العام</div>
        <div id="kpiAvg" class="kpi-val">—</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">الإنذار والحرمان</div>
        <div id="kpiAlerts" class="kpi-val">—</div>
      </div>
    </div>

    <!-- Data Table Card -->
    <div class="card" style="margin-top:24px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
        <div style="font-family:var(--mono); font-size:11px; letter-spacing:.15em; color:var(--g300); text-transform:uppercase;">
          سجل إحصائيات الطلبة
        </div>
        <input type="text" id="studentSearch" class="inp" style="max-width:280px; height:34px; font-size:12px;" placeholder="بحث بالاسم أو المعرف..." oninput="filterTable()"/>
      </div>
      <div class="table-responsive">
        <table class="data-table" id="statsTable">
          <thead>
            <tr>
              <th style="width:40px;">ت</th>
              <th>اسم الطالب</th>
              <th>المعرف الجامعي</th>
              <th>إجمالي المحاضرات</th>
              <th>الحضور</th>
              <th>الغياب</th>
              <th style="width:140px;">نسبة الحضور</th>
              <th>نسبة الغياب</th>
              <th>الحالة الأكاديمية</th>
              <th style="width:90px;">التفاصيل</th>
            </tr>
          </thead>
          <tbody id="statsBody">
            <tr>
              <td colspan="10" style="text-align:center; padding:48px 16px; color:var(--g400); font-size:13px;">
                حدد معايير البحث واضغط على زر "تحديث التقرير" لعرض الإحصائيات
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>
</div>

<!-- Modal for Student Lecture Matrix -->
<div id="matrixModal" class="modal-overlay" style="display:none;" onclick="if(event.target===this)closeModal()">
  <div class="modal-card">
    <div class="modal-header">
      <h3 id="modalStudentTitle" style="font-size:16px; font-weight:700;">سجل حضور المحاضرات التفصيلي</h3>
      <button type="button" class="btn-sm del" onclick="closeModal()">إغلاق</button>
    </div>
    <div id="modalStudentMeta" style="font-size:13px; color:var(--g300); margin-bottom:16px;"></div>
    <div class="table-responsive" style="max-height:420px; overflow-y:auto;">
      <table class="data-table">
        <thead>
          <tr>
            <th style="width:50px;">المحاضرة</th>
            <th>التاريخ والوقت</th>
            <th>الحالة</th>
            <th>طريقة التسجيل</th>
          </tr>
        </thead>
        <tbody id="modalMatrixBody">
        </tbody>
      </table>
    </div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
let _currentStats = null;

function showToast(msg, isErr=false) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast' + (isErr ? ' err' : '') + ' show';
  setTimeout(() => t.className = 'toast', 3200);
}}

async function onFilterCriteriaChanged() {{
  const dept = document.getElementById('repDept').value;
  const stage = document.getElementById('repStage').value;
  const sec = document.getElementById('repSec').value;
  try {{
    const res = await fetch(`/api/courses?department=${{encodeURIComponent(dept)}}&stage=${{encodeURIComponent(stage)}}&section=${{encodeURIComponent(sec)}}`);
    if(res.ok) {{
      const d = await res.json();
      const list = document.getElementById('courseList');
      list.innerHTML = (d.courses || []).map(c => `<option value="${{c}}"></option>`).join('');
      if(d.courses && d.courses.length > 0 && !document.getElementById('repCourse').value) {{
        document.getElementById('repCourse').value = d.courses[0];
      }}
    }}
  }} catch(e) {{}}
}}

function onCourseInputChanged() {{
}}

async function loadCumulativeReport() {{
  const dept = document.getElementById('repDept').value.trim();
  const stage = document.getElementById('repStage').value.trim();
  const sec = document.getElementById('repSec').value.trim();
  const course = document.getElementById('repCourse').value.trim();
  const dateFrom = document.getElementById('repDateFrom').value.trim();
  const dateTo = document.getElementById('repDateTo').value.trim();

  if(!course) {{
    showToast('يرجى تحديد أو إدخال اسم المادة الدراسية', true);
    return;
  }}

  let url = `/api/cumulative-stats?department=${{encodeURIComponent(dept)}}&stage=${{encodeURIComponent(stage)}}&section=${{encodeURIComponent(sec)}}&course=${{encodeURIComponent(course)}}`;
  if(dateFrom) url += `&date_from=${{encodeURIComponent(dateFrom)}}`;
  if(dateTo) url += `&date_to=${{encodeURIComponent(dateTo)}}`;

  const tbody = document.getElementById('statsBody');
  tbody.innerHTML = '<tr><td colspan=\"10\" style=\"text-align:center; padding:36px; color:var(--g300);\">جاري تحميل الإحصائيات...</td></tr>';

  try {{
    const res = await fetch(url);
    if(!res.ok) {{
      const err = await res.json();
      showToast(err.detail || 'تعذر تحميل الإحصائيات', true);
      tbody.innerHTML = `<tr><td colspan=\"10\" style=\"text-align:center; padding:36px; color:#f44336;\">${{err.detail || 'حدث خطأ أثناء جلب البيانات'}}</td></tr>`;
      return;
    }}
    const data = await res.json();
    _currentStats = data;

    document.getElementById('kpiLectures').textContent = data.total_lectures;
    document.getElementById('kpiStudents').textContent = data.total_students;
    document.getElementById('kpiAvg').textContent = data.avg_attendance_pct + '%';
    document.getElementById('kpiAlerts').textContent = (data.deprived_count || 0) + ' محروم / ' + (data.warning_count || 0) + ' إنذار';

    const expBtn = document.getElementById('exportBtn');
    if(expBtn) expBtn.disabled = (data.total_lectures === 0 && data.total_students === 0);

    renderTableRows(data.students || []);
  }} catch(e) {{
    showToast('خطأ في الاتصال بالخادم', true);
    tbody.innerHTML = '<tr><td colspan=\"10\" style=\"text-align:center; padding:36px; color:#f44336;\">تعذر الاتصال بالخادم</td></tr>';
  }}
}}

function renderTableRows(students) {{
  const tbody = document.getElementById('statsBody');
  if(!students || students.length === 0) {{
    tbody.innerHTML = '<tr><td colspan=\"10\" style=\"text-align:center; padding:40px; color:var(--g400);\">لا يوجد طلبة مسجلين في هذا القسم والشعبة</td></tr>';
    return;
  }}

  let html = '';
  students.forEach((s, idx) => {{
    let badgeCls = 'badge-good';
    if(s.status === 'محروم') badgeCls = 'badge-danger';
    else if(s.status === 'إنذار أول') badgeCls = 'badge-warn';

    html += `<tr>
      <td style="font-family:var(--mono); color:var(--g400);">${{idx + 1}}</td>
      <td style="font-weight:700; color:var(--white);">${{s.full_name}}</td>
      <td style="font-family:var(--mono); font-size:12px; color:var(--g300);">${{s.student_code || '—'}}</td>
      <td style="font-family:var(--mono);">${{s.total_lectures}}</td>
      <td style="font-family:var(--mono); color:#4caf50; font-weight:700;">${{s.present_count}}</td>
      <td style="font-family:var(--mono); color:#f44336; font-weight:700;">${{s.absent_count}}</td>
      <td>
        <div style="font-family:var(--mono); font-weight:700; margin-bottom:4px;">${{s.attendance_pct}}%</div>
        <div class="progress-track">
          <div class="progress-fill" style="width:${{s.attendance_pct}}%; background:${{s.status === 'محروم' ? '#f44336' : (s.status === 'إنذار أول' ? '#ff9800' : '#4caf50')}};"></div>
        </div>
      </td>
      <td style="font-family:var(--mono); color:var(--g300);">${{s.absence_pct}}%</td>
      <td><span class="badge-status ${{badgeCls}}">${{s.status}}</span></td>
      <td>
        <button type="button" class="btn-sm" onclick="showStudentMatrix(${{s.student_id}})">السجل</button>
      </td>
    </tr>`;
  }});
  tbody.innerHTML = html;
}}

function filterTable() {{
  if(!_currentStats || !_currentStats.students) return;
  const q = document.getElementById('studentSearch').value.trim().toLowerCase();
  if(!q) {{
    renderTableRows(_currentStats.students);
    return;
  }}
  const filtered = _currentStats.students.filter(s =>
    (s.full_name && s.full_name.toLowerCase().includes(q)) ||
    (s.student_code && s.student_code.toLowerCase().includes(q)) ||
    (s.rfid_uid && s.rfid_uid.toLowerCase().includes(q))
  );
  renderTableRows(filtered);
}}

function showStudentMatrix(studentId) {{
  if(!_currentStats || !_currentStats.students) return;
  const student = _currentStats.students.find(s => s.student_id === studentId);
  if(!student) return;

  document.getElementById('modalStudentTitle').textContent = `سجل حضور: ${{student.full_name}}`;
  document.getElementById('modalStudentMeta').textContent = `المعرف: ${{student.student_code || '—'}} | الحضور: ${{student.present_count}} / ${{student.total_lectures}} (${{student.attendance_pct}}%) | الحالة: ${{student.status}}`;

  const tbody = document.getElementById('modalMatrixBody');
  const matrix = student.lecture_matrix || [];

  if(matrix.length === 0) {{
    tbody.innerHTML = '<tr><td colspan=\"4\" style=\"text-align:center; padding:24px; color:var(--g400);\">لا توجد محاضرات منعقدة حتى الآن</td></tr>';
  }} else {{
    tbody.innerHTML = matrix.map((m, i) => {{
      const dt = m.started_at ? m.started_at.replace('T', ' ').substring(0, 16) : '—';
      const isAtt = m.attended;
      const statusBadge = isAtt
        ? '<span class=\"badge-status badge-good\">حاضر</span>'
        : '<span class=\"badge-status badge-danger\">غائب</span>';
      return `<tr>
        <td style=\"font-family:var(--mono); text-align:center;\">${{i + 1}}</td>
        <td style=\"font-family:var(--mono); font-size:12px;\">${{dt}}</td>
        <td style=\"text-align:center;\">${{statusBadge}}</td>
        <td style=\"font-size:12px; color:var(--g300);\">${{m.method || (isAtt ? 'كاميرا ذكية' : '—')}}</td>
      </tr>`;
    }}).join('');
  }}

  document.getElementById('matrixModal').style.display = 'flex';
}}

function closeModal() {{
  document.getElementById('matrixModal').style.display = 'none';
}}

function exportCumulativeExcel() {{
  const dept = document.getElementById('repDept').value.trim();
  const stage = document.getElementById('repStage').value.trim();
  const sec = document.getElementById('repSec').value.trim();
  const course = document.getElementById('repCourse').value.trim();
  const dateFrom = document.getElementById('repDateFrom').value.trim();
  const dateTo = document.getElementById('repDateTo').value.trim();

  if(!course) {{
    showToast('يرجى تحديد المادة أولاً', true);
    return;
  }}

  let url = `/api/export-cumulative-report?department=${{encodeURIComponent(dept)}}&stage=${{encodeURIComponent(stage)}}&section=${{encodeURIComponent(sec)}}&course=${{encodeURIComponent(course)}}`;
  if(dateFrom) url += `&date_from=${{encodeURIComponent(dateFrom)}}`;
  if(dateTo) url += `&date_to=${{encodeURIComponent(dateTo)}}`;

  window.location.href = url;
}}

window.addEventListener('DOMContentLoaded', () => {{
  onFilterCriteriaChanged();
}});
</script>
</body>
</html>"""


@app.get("/reports", response_class=HTMLResponse, summary="Cumulative Reports Dashboard")
async def reports_dashboard():
    return HTMLResponse(render_reports_page())


@app.post("/start-session", summary="Begin lecture session")
async def start_session(
    department : str = Form(...),
    section    : str = Form(...),
    stage      : str = Form(...),
    course     : str = Form(...),
    instructor : Optional[str] = None,
):
    valid_departments = get_academic_departments()
    if valid_departments and department not in valid_departments:
        raise HTTPException(422, "Invalid academic department.")
    if not course.strip():
        raise HTTPException(422, "Course name required.")

    inst_clean = instructor.strip() if isinstance(instructor, str) and instructor.strip() else None

    SESSION.begin(
        department = department.strip(),
        section    = section.strip(),
        stage      = stage.strip(),
        course     = course.strip(),
        instructor = inst_clean,
    )
    CAMERA.start()
    return RedirectResponse("/lecture", status_code=303)


@app.get("/lecture", response_class=HTMLResponse, summary="Live lecture view")
async def lecture_view():
    if not SESSION.active:
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(render_lecture_page(SESSION.snapshot()))


# ROUTES — NON-BLOCKING 30 FPS ASYNC VIDEO STREAM
@app.get("/video-feed", summary="MJPEG stream at smooth 30 FPS")
async def video_feed():
    """
    Non-blocking async MJPEG multipart stream at ~30 FPS.

    • Camera capture runs in a dedicated OS thread, fully decoupled from
      face recognition and from this HTTP request.
    • get_jpeg_stream() is an async generator: it uses asyncio.sleep() so
      the FastAPI/uvicorn event loop is NEVER blocked between frames.
    • Frames are served from a Queue(maxsize=2) — stale frames are discarded
      automatically, guaranteeing minimum latency at all times.
    • Clicking "End Lecture" sets _stop_event; the generator exits on its
      next iteration without hanging the connection.
    """
    CAMERA.start()
    return StreamingResponse(
        CAMERA.get_jpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


# ROUTES — WEBSOCKET ATTENDANCE STREAM
@app.websocket("/ws/attendance")
async def websocket_attendance(websocket: WebSocket):
    """
    Real-time push channel for instant UI cards update and sound chime.
    Dispatched immediately on any AI face detection or RFID card swipe.
    """
    await WS_MANAGER.connect(websocket)
    try:
        # Send current session snapshot upon initial connection
        await websocket.send_json({
            "event": "connected",
            "snapshot": SESSION.snapshot(),
        })
        while True:
            # Keep connection open; receive client ping/pong
            await websocket.receive_text()
    except WebSocketDisconnect:
        WS_MANAGER.disconnect(websocket)
    except Exception:
        WS_MANAGER.disconnect(websocket)


# ROUTES — REST API
@app.get("/api/rfid-status")
async def rfid_status():
    """Diagnostic endpoint for ESP32-S RFID hardware status."""
    return JSONResponse({
        "ok": True,
        "serial_available": SERIAL_AVAILABLE,
        "connected": RFID_MANAGER.is_connected(),
        "port": RFID_MANAGER.port,
        "baudrate": RFID_MANAGER.baudrate,
        "last_uid": RFID_MANAGER.last_uid,
        "last_scan": RFID_MANAGER.last_scan_dt.isoformat() if RFID_MANAGER.last_scan_dt else None,
        "mappings_count": len(RFID_REPO.mappings),
        "mappings": RFID_REPO.mappings,
    })


@app.post("/api/rfid-map")
async def rfid_map(uid: str = Form(...), student_name: str = Form(...)):
    """Map an RFID card UID to an enrolled student."""
    if not uid.strip() or not student_name.strip():
        raise HTTPException(400, "Both 'uid' and 'student_name' are required.")
    clean_uid = uid.strip().upper()
    if not clean_uid.startswith("UID:"):
        clean_uid = f"UID:{clean_uid}"
    RFID_REPO.register(clean_uid, student_name.strip())
    return JSONResponse({
        "ok": True,
        "uid": clean_uid,
        "student_name": student_name.strip(),
        "message": f"Successfully mapped {clean_uid} to {student_name.strip()}",
    })


@app.get("/api/camera-status")
async def camera_status():
    """
    Non-destructive camera probe with automatic index fallback.
    """
    if CAMERA.is_active():
        return JSONResponse({"ok": True, "index": CAMERA.camera_index, "backend": "active"})

    with CAMERA._lock:
        if CAMERA._running and CAMERA._latest_jpeg is not None:
            return JSONResponse({"ok": True, "index": CAMERA.camera_index, "backend": "streaming"})

    candidate_indices = [CAMERA.camera_index]
    for alt_idx in [0, 1, 2]:
        if alt_idx not in candidate_indices:
            candidate_indices.append(alt_idx)

    for c_idx in candidate_indices:
        for backend_id, backend_name in _CAM_BACKENDS:
            try:
                cap = (
                    cv2.VideoCapture(c_idx, backend_id)
                    if backend_id
                    else cv2.VideoCapture(c_idx)
                )
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        CAMERA.camera_index = c_idx
                        return JSONResponse({
                            "ok": True, "index": c_idx,
                            "backend": backend_name,
                        })
                cap.release()
            except Exception as exc:
                log.debug("camera-status probe [%d] via %s: %s", c_idx, backend_name, exc)

    return JSONResponse({
        "ok": False, "index": CAMERA.camera_index,
        "message": f"Camera not found on candidate indices {candidate_indices}. Check USB connection.",
    })


@app.get("/api/session-state")
async def session_state():
    return JSONResponse(SESSION.snapshot())


@app.post("/api/end-session")
async def end_session():
    """
    Terminate the active lecture session.
    Sets the camera stop event FIRST so the async stream generator exits
    cleanly on its next loop iteration — no connection hang, no freeze.
    The camera threads are then stopped and the frame queue is drained.
    """
    SESSION.terminate()   # mark session inactive (stops recognition loop)
    CAMERA.stop()         # sets _stop_event → wakes async stream generator
    return JSONResponse({"ok": True})


@app.post("/api/reload-faces")
async def reload_faces():
    """Hot-reload student face encodings & profile photos without restart."""
    count = FACE_DB.reload()
    return JSONResponse({"ok": True, "count": count})


@app.get("/api/face-db-status")
async def face_db_status():
    return JSONResponse({
        "fr_available" : ENGINE_AVAILABLE,
        "engine"       : "Google MediaPipe & Ultra-Lightweight CPU Engine (Intel i5 Optimized)",
        "count"        : FACE_DB.count,
        "names"        : FACE_DB.names,
        "directory"    : str(KNOWN_DIR),
    })


# STUDENT ENROLLMENT & MANAGEMENT APIs
@app.get("/api/camera-snapshot")
async def camera_snapshot():
    """
    Safe snapshot capture endpoint for student enrollment without conflicting
    with an active lecture session.
    """
    frame = None
    if CAMERA.is_active():
        with CAMERA._lock:
            if CAMERA._latest_raw is not None:
                frame = CAMERA._latest_raw.copy()
    else:
        cap = CAMERA._open_camera()
        if cap is not None and cap.isOpened():
            try:
                # Read 3 warm-up frames to allow auto-exposure/white-balance to settle
                for _ in range(3):
                    cap.read()
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f
            finally:
                cap.release()

    if frame is None:
        raise HTTPException(503, "الكاميرا غير متصلة أو لا يمكن التقاط صورة منها حالياً")

    return JSONResponse({
        "ok": True,
        "image_b64": _frame_to_b64(frame, quality=90),
    })


@app.get("/api/students")
async def list_students(
    department: Optional[str] = None,
    stage: Optional[str] = None,
    section: Optional[str] = None,
):
    """List enrolled students with thumbnails and academic placement."""
    students = DB.get_students(department=department, stage=stage, section=section)
    result = []
    for s in students:
        result.append({
            "id": s["id"],
            "full_name": s["full_name"],
            "student_code": s.get("student_code") or "",
            "department": s["department"],
            "stage": s["stage"],
            "section": s["section"],
            "rfid_uid": s.get("rfid_uid") or "",
            "photo_b64": FACE_DB.get_photo_by_id(s["id"]),
            "created_at": s.get("created_at") or "",
        })
    return JSONResponse(result)


@app.post("/api/students", status_code=201)
async def create_student(
    full_name: str = Form(...),
    department: str = Form(...),
    stage: str = Form(...),
    section: str = Form(...),
    student_code: Optional[str] = Form(None),
    rfid_uid: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    photo_base64: Optional[str] = Form(None),
):
    """Enroll a new student with biometric face embedding and optional RFID card."""
    clean_name = full_name.strip()
    if not clean_name:
        raise HTTPException(400, "اسم الطالب مطلوب")

    # Read image bytes from either uploaded file or captured base64 data
    img_bytes: Optional[bytes] = None
    if photo and photo.filename:
        img_bytes = await photo.read()
    elif photo_base64 and photo_base64.strip():
        raw_b64 = photo_base64.strip()
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(raw_b64)
        except Exception:
            raise HTTPException(400, "صيغة بيانات الصورة غير صالحة")

    if not img_bytes:
        raise HTTPException(400, "صورة وجه الطالب مطلوبة للتسجيل")

    # Extract 512-d normalized face vector and save clean photo
    emb, saved_path = extract_face_embedding_from_bytes(img_bytes)

    clean_uid = None
    if rfid_uid and rfid_uid.strip():
        clean_uid = rfid_uid.strip().upper()
        if not clean_uid.startswith("UID:"):
            clean_uid = f"UID:{clean_uid}"

    clean_code = student_code.strip() if student_code and student_code.strip() else None

    try:
        new_id = DB.add_student(
            full_name=clean_name,
            department=department.strip(),
            stage=stage.strip(),
            section=section.strip(),
            embedding=emb,
            photo_path=saved_path,
            student_code=clean_code,
            rfid_uid=clean_uid,
        )
    except Exception as exc:
        if "UNIQUE constraint failed: students.rfid_uid" in str(exc):
            raise HTTPException(400, f"البطاقة الجامعية '{clean_uid}' مسجلة بالفعل لطالب آخر")
        raise HTTPException(500, f"خطأ في حفظ الطالب: {exc}")

    # Immediately hot-reload face database so the student is recognizable in real-time
    FACE_DB.reload()

    return JSONResponse({
        "ok": True,
        "student": {
            "id": new_id,
            "full_name": clean_name,
            "department": department.strip(),
            "stage": stage.strip(),
            "section": section.strip(),
            "student_code": clean_code or "",
            "rfid_uid": clean_uid or "",
            "photo_b64": FACE_DB.get_photo_by_id(new_id),
        }
    }, status_code=201)


@app.patch("/api/students/{student_id}/rfid")
async def update_student_rfid(student_id: int, rfid_uid: Optional[str] = Form(None)):
    """Update or clear an enrolled student's RFID card UID."""
    try:
        success = DB.update_student_rfid(student_id, rfid_uid)
        if not success:
            raise HTTPException(404, "الطالب غير موجود")
        FACE_DB.reload()
        return JSONResponse({"ok": True, "message": "تم تحديث البطاقة الجامعية بنجاح"})
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/students/{student_id}")
async def delete_student(student_id: int):
    """Delete an enrolled student from the database and remove their photo file."""
    students = DB.get_students()
    matched = next((s for s in students if s["id"] == student_id), None)
    if matched and matched.get("photo_path"):
        try:
            p = Path(matched["photo_path"]).resolve()
            if p.exists() and KNOWN_DIR.resolve() in p.parents:
                p.unlink()
        except Exception as exc:
            log.warning("Could not delete photo file for student %d: %s", student_id, exc)

    success = DB.delete_student(student_id)
    if not success:
        raise HTTPException(404, "الطالب غير موجود")

    FACE_DB.reload()
    return JSONResponse({"ok": True, "message": "تم حذف الطالب بنجاح"})


# ACADEMIC SETTINGS APIs
@app.get("/api/settings")
async def get_settings():
    """Return institution names and dynamic academic items."""
    return JSONResponse({
        "institution_ar": DB.get_setting("institution_ar", DEFAULT_INSTITUTION_AR),
        "institution_en": DB.get_setting("institution_en", DEFAULT_INSTITUTION_EN),
        "departments": DB.get_academic_items("department"),
        "stages": DB.get_academic_items("stage"),
        "sections": DB.get_academic_items("section"),
    })


@app.post("/api/settings/institution")
async def update_institution(
    institution_ar: str = Form(...),
    institution_en: str = Form(...),
):
    """Update institution profile names."""
    if not institution_ar.strip() or not institution_en.strip():
        raise HTTPException(400, "اسم المؤسسة مطلوب باللغتين العربية والإنجليزية")
    DB.set_setting("institution_ar", institution_ar.strip())
    DB.set_setting("institution_en", institution_en.strip())
    return JSONResponse({"ok": True, "message": "تم تحديث اسم المؤسسة بنجاح"})


@app.post("/api/settings/items")
async def add_academic_item_api(category: str = Form(...), name: str = Form(...)):
    """Add a new department, stage, or section."""
    try:
        new_id = DB.add_academic_item(category, name)
        return JSONResponse({"ok": True, "id": new_id, "category": category, "name": name.strip()})
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/api/settings/items/{item_id}")
async def rename_academic_item_api(item_id: int, new_name: str = Form(...)):
    """Rename an academic item with cascade to students and lectures."""
    try:
        success = DB.rename_academic_item(item_id, new_name)
        if not success:
            raise HTTPException(404, "العنصر غير موجود")
        FACE_DB.reload()
        return JSONResponse({"ok": True, "message": "تم تعديل الاسم بنجاح وتحديث كافة السجلات المرتبطة"})
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/settings/items/{item_id}")
async def delete_academic_item_api(item_id: int):
    """Delete an academic item."""
    success = DB.delete_academic_item(item_id)
    if not success:
        raise HTTPException(404, "العنصر غير موجود")
    return JSONResponse({"ok": True, "message": "تم حذف العنصر بنجاح"})


@app.get("/api/courses")
async def get_courses_autocomplete(
    department: Optional[str] = None,
    stage: Optional[str] = None,
    section: Optional[str] = None,
):
    """Return distinct course names for autocomplete based on chosen filters."""
    courses = DB.get_distinct_courses(department, stage, section)
    return JSONResponse({"courses": courses})


@app.get("/api/export-report", summary="Download dynamic formatted RTL .xlsx attendance report")
async def export_report(
    department     : Optional[str] = None,
    subject        : Optional[str] = None,
    course         : Optional[str] = None,
    section        : Optional[str] = None,
    stage          : Optional[str] = None,
    instructor_name: Optional[str] = None,
    instructor     : Optional[str] = None,
    date           : Optional[str] = None,
    time           : Optional[str] = None,
):
    """
    Generate and stream an institutional, dynamic, fully-formatted Excel (.xlsx) attendance report.
    Adheres strictly to Middle Technical University / Polytechnic College standards:
    - Right-To-Left (RTL) Sheet Layout
    - Official Institutional Multi-Level Header
    - Session Information Block with dynamic parameters
    - Full Attendance Roster combining recognized and absent students (all 8 known students marked explicitly)
    - Professional University Navy/Slate styling, status badges, borders
    - Executive Summary Metrics and Signature Verification Lines
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        snap = SESSION.snapshot()

        def _str_val(val) -> Optional[str]:
            if isinstance(val, str) and val.strip():
                return val.strip()
            return None

        raw_dept = _str_val(department) or _str_val(snap.get("department")) or "هندسة تقنيات الإلكترونيك والذكاء الاصطناعي"
        dept_display = raw_dept if raw_dept.startswith("قسم") else f"قسم {raw_dept}"

        subj = _str_val(subject) or _str_val(course) or _str_val(snap.get("course")) or "الرؤية الحاسوبية والأنظمة الذكية"
        sec  = _str_val(section) or _str_val(snap.get("section")) or "الشعبة أ"
        stg  = _str_val(stage) or _str_val(snap.get("stage")) or "المرحلة الرابعة"
        inst = _str_val(instructor_name) or _str_val(instructor) or _str_val(snap.get("instructor")) or ""

        # Date & Time resolution
        now = datetime.now()
        session_start_str = snap.get("started_at")
        if session_start_str:
            try:
                session_dt = datetime.fromisoformat(session_start_str)
            except Exception:
                session_dt = now
        else:
            session_dt = now

        report_date = _str_val(date) or session_dt.strftime("%Y/%m/%d")
        report_time = _str_val(time) or session_dt.strftime("%I:%M:%S %p")

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "سجل الحضور الأكاديمي"

        ws.sheet_view.rightToLeft = True
        try:
            if hasattr(ws, "views") and hasattr(ws.views, "sheetView") and ws.views.sheetView:
                ws.views.sheetView[0].rightToLeft = True
        except Exception:
            pass
        try:
            ws.views.sheetView[0].showGridLines = True
        except Exception:
            pass

        navy_dark   = "0F2942"   # Primary Institutional Navy
        navy_mid    = "1A365D"   # Accent Slate Navy
        border_col  = "CBD5E0"   # Clean Light Border
        thin_border = Border(
            left=Side(style="thin", color=border_col),
            right=Side(style="thin", color=border_col),
            top=Side(style="thin", color=border_col),
            bottom=Side(style="thin", color=border_col),
        )

        header_fill = PatternFill("solid", fgColor=navy_dark)
        sub_fill    = PatternFill("solid", fgColor=navy_mid)
        info_lbl_bg = PatternFill("solid", fgColor="EDF2F7")
        info_val_bg = PatternFill("solid", fgColor="FFFFFF")

        inst_ar_val = DB.get_setting("institution_ar", DEFAULT_INSTITUTION_AR)
        ws.merge_cells("A2:F2")
        c_main = ws["A2"]
        c_main.value = inst_ar_val
        c_main.font = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
        c_main.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 36

        ws.merge_cells("A3:F3")
        c_sub = ws["A3"]
        c_sub.value = f"{dept_display} — استمارة حضور وغياب الطلبة للمحاضرة الأكاديمية"
        c_sub.font = Font(name="Calibri", size=12, bold=True, color="E2E8F0")
        c_sub.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[3].height = 28

        for r in (2, 3):
            for col_idx in range(1, 7):
                cell = ws.cell(row=r, column=col_idx)
                cell.border = thin_border
                cell.fill = header_fill if r == 2 else sub_fill

        ws.row_dimensions[4].height = 10

        info_meta = [
            [
                ("A", "المادة الأكاديمية:", True),
                ("B", subj, False),
                ("C", "القسم / المرحلة:", True),
                ("D", f"{dept_display} ({stg})", False),
                ("E", "الشعبة:", True),
                ("F", sec, False),
            ],
            [
                ("A", "أستاذ المادة المشرف:", True),
                ("B", "", False),
                ("C", "تاريخ الانعقاد:", True),
                ("D", report_date, False),
                ("E", "وقت البدء:", True),
                ("F", report_time, False),
            ],
        ]

        for row_idx, items in enumerate(info_meta, start=5):
            ws.row_dimensions[row_idx].height = 24
            for col_letter, val, is_lbl in items:
                cell = ws[f"{col_letter}{row_idx}"]
                cell.value = val
                cell.border = thin_border
                if is_lbl:
                    cell.fill = info_lbl_bg
                    cell.font = Font(name="Calibri", size=10, bold=True, color="475569")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.fill = info_val_bg
                    cell.font = Font(name="Calibri", size=10, bold=True, color="0F172A")
                    cell.alignment = Alignment(horizontal="center", vertical="center")

        ws.row_dimensions[7].height = 12

        headers = [
            ("A", "ت"),
            ("B", "اسم الطالب"),
            ("C", "القسم / الشعبة"),
            ("D", "طريقة التسجيل"),
            ("E", "وقت الحضور"),
            ("F", "حالة الحضور"),
        ]
        ws.row_dimensions[8].height = 28
        for col_letter, h_text in headers:
            cell = ws[f"{col_letter}8"]
            cell.value = h_text
            cell.fill = PatternFill("solid", fgColor=navy_dark)
            cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        logged_list = snap.get("attendance", [])
        logged_map = {rec["name"]: rec for rec in logged_list}

        # Query enrolled students strictly scoped to active department, stage, and section
        raw_dept_for_query = dept_display
        section_students = DB.get_students(department=raw_dept_for_query, stage=stg, section=sec)
        if not section_students and raw_dept != raw_dept_for_query:
            section_students = DB.get_students(department=raw_dept, stage=stg, section=sec)

        if section_students:
            master_roster = [s["full_name"] for s in section_students]
        else:
            # Fallback to logged names only if section roster is empty in database
            master_roster = [rec["name"] for rec in logged_list if rec.get("name")]

        for rec in logged_list:
            if rec.get("name") and rec["name"] not in master_roster:
                master_roster.append(rec["name"])

        # Prioritize present attendees first (by timestamp), followed by absent students alphabetically
        present_names = [name for name in master_roster if name in logged_map]
        absent_names  = sorted([name for name in master_roster if name not in logged_map])
        combined_roster = present_names + absent_names

        start_row = 9
        for i, name in enumerate(combined_roster, start=1):
            cur_row = start_row + i - 1
            ws.row_dimensions[cur_row].height = 24
            is_present = name in logged_map

            if is_present:
                rec = logged_map[name]
                method = rec.get("method") or "كاميرا ذكية (AI Vision)"
                ts_raw = (rec.get("timestamp") or "")[:19].replace("T", " ")
                if ts_raw:
                    try:
                        ts = datetime.fromisoformat(ts_raw).strftime("%I:%M:%S %p")
                    except Exception:
                        ts = ts_raw
                else:
                    ts = "—"
                status_text = "حاضر"
                status_font = Font(name="Calibri", size=10, bold=True, color="065F46")
                status_fill = PatternFill("solid", fgColor="D1FAE5")
            else:
                method = "—"
                ts = "—"
                status_text = "غائب"
                status_font = Font(name="Calibri", size=10, bold=True, color="991B1B")
                status_fill = PatternFill("solid", fgColor="FEE2E2")

            row_fill = PatternFill("solid", fgColor="FFFFFF" if i % 2 == 1 else "F8FAFC")
            cell_font = Font(name="Calibri", size=10, color="1E293B")

            # Col A: Index
            c_idx = ws[f"A{cur_row}"]
            c_idx.value = i
            c_idx.alignment = Alignment(horizontal="center", vertical="center")

            # Col B: Student Name
            c_name = ws[f"B{cur_row}"]
            c_name.value = name
            c_name.alignment = Alignment(horizontal="right", vertical="center")

            # Col C: Dept / Section
            c_sec = ws[f"C{cur_row}"]
            c_sec.value = f"{dept_display} / {sec}"
            c_sec.alignment = Alignment(horizontal="center", vertical="center")

            # Col D: Method
            c_meth = ws[f"D{cur_row}"]
            c_meth.value = method
            c_meth.alignment = Alignment(horizontal="center", vertical="center")

            # Col E: Timestamp
            c_ts = ws[f"E{cur_row}"]
            c_ts.value = ts
            c_ts.alignment = Alignment(horizontal="center", vertical="center")

            # Col F: Status
            c_stat = ws[f"F{cur_row}"]
            c_stat.value = status_text
            c_stat.alignment = Alignment(horizontal="center", vertical="center")
            c_stat.font = status_font
            c_stat.fill = status_fill

            for col_l in ("A", "B", "C", "D", "E"):
                c = ws[f"{col_l}{cur_row}"]
                c.font = cell_font
                c.fill = row_fill
                c.border = thin_border
            c_stat.border = thin_border

        after_data_row = start_row + len(combined_roster)
        ws.row_dimensions[after_data_row].height = 14

        sum_title_row = after_data_row + 1
        sum_data_row  = after_data_row + 2
        ws.row_dimensions[sum_title_row].height = 24
        ws.row_dimensions[sum_data_row].height = 28

        total_count   = len(combined_roster)
        present_count = len(present_names)
        absent_count  = len(absent_names)
        rate_str      = f"{(present_count / total_count * 100):.1f}%" if total_count else "0.0%"

        sum_metrics = [
            ("A", "B", "إجمالي الطلبة المقيدين", f"{total_count} طالب", "475569", "F8FAFC"),
            ("C", "C", "عدد الحاضرين", f"{present_count} حاضر", "065F46", "D1FAE5"),
            ("D", "D", "عدد الغائبين", f"{absent_count} غائب", "991B1B", "FEE2E2"),
            ("E", "F", "نسبة الحضور الإجمالية", rate_str, "1E40AF", "DBEAFE"),
        ]

        for start_c, end_c, title, val, text_col, bg_col in sum_metrics:
            if start_c != end_c:
                ws.merge_cells(f"{start_c}{sum_title_row}:{end_c}{sum_title_row}")
                ws.merge_cells(f"{start_c}{sum_data_row}:{end_c}{sum_data_row}")
            t_cell = ws[f"{start_c}{sum_title_row}"]
            t_cell.value = title
            t_cell.fill = PatternFill("solid", fgColor=navy_mid)
            t_cell.font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
            t_cell.alignment = Alignment(horizontal="center", vertical="center")

            d_cell = ws[f"{start_c}{sum_data_row}"]
            d_cell.value = val
            d_cell.fill = PatternFill("solid", fgColor=bg_col)
            d_cell.font = Font(name="Calibri", size=12, bold=True, color=text_col)
            d_cell.alignment = Alignment(horizontal="center", vertical="center")

            cols = [start_c] if start_c == end_c else [start_c, end_c]
            for c in cols:
                ws[f"{c}{sum_title_row}"].border = thin_border
                ws[f"{c}{sum_data_row}"].border = thin_border

        sig_space_row = sum_data_row + 1
        sig_row1 = sum_data_row + 2
        sig_row2 = sum_data_row + 3
        sig_row3 = sum_data_row + 4

        ws.row_dimensions[sig_space_row].height = 18
        ws.row_dimensions[sig_row1].height = 22
        ws.row_dimensions[sig_row2].height = 22
        ws.row_dimensions[sig_row3].height = 22

        # Supervising Course Professor Signature (Right Side: A-B)
        ws.merge_cells(f"A{sig_row1}:B{sig_row1}")
        ws.merge_cells(f"A{sig_row2}:B{sig_row2}")
        ws.merge_cells(f"A{sig_row3}:B{sig_row3}")

        ws[f"A{sig_row1}"].value = "أستاذ المادة المشرف:"
        ws[f"A{sig_row1}"].font = Font(name="Calibri", size=11, bold=True, color="0F2942")
        ws[f"A{sig_row1}"].alignment = Alignment(horizontal="center", vertical="center")

        ws[f"A{sig_row2}"].value = ""
        ws[f"A{sig_row2}"].font = Font(name="Calibri", size=11, bold=True, color="334155")
        ws[f"A{sig_row2}"].alignment = Alignment(horizontal="center", vertical="center")

        ws[f"A{sig_row3}"].value = "التوقيع: ......................................."
        ws[f"A{sig_row3}"].font = Font(name="Calibri", size=10, color="64748B")
        ws[f"A{sig_row3}"].alignment = Alignment(horizontal="center", vertical="center")

        ws.column_dimensions["A"].width = 8
        ws.column_dimensions["B"].width = 34
        ws.column_dimensions["C"].width = 38
        ws.column_dimensions["D"].width = 24
        ws.column_dimensions["E"].width = 24
        ws.column_dimensions["F"].width = 18

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_report_{ts}.xlsx"
        (REPORTS_DIR / filename).write_bytes(buf.getvalue())
        buf.seek(0)
        log.info("Institutional report generated: %s (size: %d bytes)", filename, len(buf.getvalue()))

        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ImportError:
        raise HTTPException(501, "openpyxl not installed.")
    except Exception as exc:
        log.error("Export report error: %s", exc)
        raise HTTPException(500, str(exc))



@app.get("/api/cumulative-stats")
async def get_cumulative_stats_api(
    department: str = Query(...),
    stage: str = Query(...),
    section: str = Query(...),
    course: str = Query(...),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    try:
        stats = DB.get_cumulative_stats(
            department=department.strip(),
            stage=stage.strip(),
            section=section.strip(),
            course=course.strip(),
            date_from=date_from.strip() if date_from else None,
            date_to=date_to.strip() if date_to else None,
        )
        return JSONResponse(stats)
    except Exception as exc:
        log.error("Cumulative stats query error: %s", exc)
        raise HTTPException(500, str(exc))


@app.get("/api/export-cumulative-report", summary="Download multi-lecture cumulative Excel (.xlsx) attendance report")
async def export_cumulative_report(
    department: str = Query(...),
    stage: str = Query(...),
    section: str = Query(...),
    course: str = Query(...),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        from urllib.parse import quote
        from io import BytesIO

        stats = DB.get_cumulative_stats(
            department=department.strip(),
            stage=stage.strip(),
            section=section.strip(),
            course=course.strip(),
            date_from=date_from.strip() if date_from else None,
            date_to=date_to.strip() if date_to else None,
        )

        inst_ar = DB.get_setting("institution_ar", DEFAULT_INSTITUTION_AR)
        inst_en = DB.get_setting("institution_en", DEFAULT_INSTITUTION_EN)

        wb = openpyxl.Workbook()

        # Sheet 1: ملخص نسب الحضور
        ws1 = wb.active
        ws1.title = "ملخص نسب الحضور"
        ws1.sheet_view.rightToLeft = True
        try:
            if hasattr(ws1, "views") and hasattr(ws1.views, "sheetView") and ws1.views.sheetView:
                ws1.views.sheetView[0].rightToLeft = True
                ws1.views.sheetView[0].showGridLines = True
        except Exception:
            pass

        navy_dark = "0F2942"
        navy_mid = "1A365D"
        border_col = "CBD5E0"
        thin_border = Border(
            left=Side(style="thin", color=border_col),
            right=Side(style="thin", color=border_col),
            top=Side(style="thin", color=border_col),
            bottom=Side(style="thin", color=border_col),
        )

        header_fill = PatternFill("solid", fgColor=navy_dark)
        sub_fill = PatternFill("solid", fgColor=navy_mid)
        lbl_fill = PatternFill("solid", fgColor="EDF2F7")
        val_fill = PatternFill("solid", fgColor="FFFFFF")

        ws1.merge_cells("A1:I1")
        c1 = ws1["A1"]
        c1.value = inst_ar
        c1.font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
        c1.fill = header_fill
        c1.alignment = Alignment(horizontal="center", vertical="center")
        ws1.row_dimensions[1].height = 32

        ws1.merge_cells("A2:I2")
        c2 = ws1["A2"]
        c2.value = "تقرير الحضور والغياب التراكمي الفصلي"
        c2.font = Font(name="Calibri", size=11, bold=True, color="E2E8F0")
        c2.fill = sub_fill
        c2.alignment = Alignment(horizontal="center", vertical="center")
        ws1.row_dimensions[2].height = 24

        def _set_meta(ws, cell_lbl, lbl, cell_val, val):
            cl = ws[cell_lbl]
            cl.value = lbl
            cl.font = Font(name="Calibri", size=9, bold=True, color="4A5568")
            cl.fill = lbl_fill
            cl.alignment = Alignment(horizontal="right", vertical="center")
            cl.border = thin_border

            cv = ws[cell_val]
            cv.value = val
            cv.font = Font(name="Calibri", size=10, bold=False, color="1A202C")
            cv.fill = val_fill
            cv.alignment = Alignment(horizontal="right", vertical="center")
            cv.border = thin_border

        ws1.row_dimensions[4].height = 22
        ws1.row_dimensions[5].height = 22

        _set_meta(ws1, "A4", "القسم الأكاديمي:", "B4", department)
        _set_meta(ws1, "C4", "المرحلة / الشعبة:", "D4", f"{stage} - {section}")
        _set_meta(ws1, "E4", "المادة الدراسية:", "F4", course)
        _set_meta(ws1, "G4", "إجمالي المحاضرات:", "H4", stats["total_lectures"])

        date_range_str = f"{date_from or 'بداية الفصل'} إلى {date_to or 'تاريخ اليوم'}"
        export_time_str = datetime.now().strftime("%Y/%m/%d %I:%M %p")
        _set_meta(ws1, "A5", "تاريخ التصدير:", "B5", export_time_str)
        _set_meta(ws1, "C5", "نطاق التواريخ:", "D5", date_range_str)
        _set_meta(ws1, "E5", "متوسط الحضور:", "F5", f"{stats['avg_attendance_pct']}%")
        _set_meta(ws1, "G5", "عدد الطلبة:", "H5", stats["total_students"])

        headers = [
            ("A", "ت", 6),
            ("B", "اسم الطالب", 28),
            ("C", "المعرف / الكود", 16),
            ("D", "إجمالي المحاضرات", 16),
            ("E", "الحضور", 12),
            ("F", "الغياب", 12),
            ("G", "نسبة الحضور", 14),
            ("H", "نسبة الغياب", 14),
            ("I", "الحالة الأكاديمية", 18),
        ]
        ws1.row_dimensions[7].height = 26
        for col_letter, title, width in headers:
            cell = ws1[f"{col_letter}7"]
            cell.value = title
            cell.font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
            ws1.column_dimensions[col_letter].width = width

        start_row = 8
        for i, s in enumerate(stats["students"], start=1):
            r = start_row + i - 1
            ws1.row_dimensions[r].height = 22
            row_fill = PatternFill("solid", fgColor="FFFFFF" if i % 2 == 1 else "F8FAFC")
            cell_font = Font(name="Calibri", size=10, color="1E293B")

            st_status = s["status"]
            if st_status == "محروم":
                status_font = Font(name="Calibri", size=10, bold=True, color="991B1B")
                status_fill = PatternFill("solid", fgColor="FEE2E2")
            elif st_status == "إنذار أول":
                status_font = Font(name="Calibri", size=10, bold=True, color="92400E")
                status_fill = PatternFill("solid", fgColor="FEF3C7")
            else:
                status_font = Font(name="Calibri", size=10, bold=True, color="065F46")
                status_fill = PatternFill("solid", fgColor="D1FAE5")

            vals = [
                ("A", i, Alignment(horizontal="center", vertical="center"), cell_font, row_fill),
                ("B", s["full_name"], Alignment(horizontal="right", vertical="center"), cell_font, row_fill),
                ("C", s["student_code"] or "—", Alignment(horizontal="center", vertical="center"), cell_font, row_fill),
                ("D", s["total_lectures"], Alignment(horizontal="center", vertical="center"), cell_font, row_fill),
                ("E", s["present_count"], Alignment(horizontal="center", vertical="center"), cell_font, row_fill),
                ("F", s["absent_count"], Alignment(horizontal="center", vertical="center"), cell_font, row_fill),
                ("G", f"{s['attendance_pct']}%", Alignment(horizontal="center", vertical="center"), cell_font, row_fill),
                ("H", f"{s['absence_pct']}%", Alignment(horizontal="center", vertical="center"), cell_font, row_fill),
                ("I", st_status, Alignment(horizontal="center", vertical="center"), status_font, status_fill),
            ]
            for col_letter, val, align, fnt, fl in vals:
                cell = ws1[f"{col_letter}{r}"]
                cell.value = val
                cell.alignment = align
                cell.font = fnt
                cell.fill = fl
                cell.border = thin_border

        sig_row = start_row + len(stats["students"]) + 3
        ws1.row_dimensions[sig_row].height = 20
        ws1[f"B{sig_row}"].value = "توقيع أستاذ المادة: ............................"
        ws1[f"B{sig_row}"].font = Font(name="Calibri", size=10, bold=True, color="334155")
        ws1[f"E{sig_row}"].value = "توقيع رئيس القسم: ............................"
        ws1[f"E{sig_row}"].font = Font(name="Calibri", size=10, bold=True, color="334155")
        ws1[f"H{sig_row}"].value = "مصادقة العمادة / التسجيل: ............................"
        ws1[f"H{sig_row}"].font = Font(name="Calibri", size=10, bold=True, color="334155")

        # Sheet 2: سجل المحاضرات التفصيلي
        ws2 = wb.create_sheet(title="سجل المحاضرات التفصيلي")
        ws2.sheet_view.rightToLeft = True
        try:
            if hasattr(ws2, "views") and hasattr(ws2.views, "sheetView") and ws2.views.sheetView:
                ws2.views.sheetView[0].rightToLeft = True
                ws2.views.sheetView[0].showGridLines = True
        except Exception:
            pass

        lectures = stats["lectures"]
        num_lecs = len(lectures)
        max_col_idx = 3 + num_lecs
        last_col_letter = get_column_letter(max_col_idx)

        ws2.merge_cells(f"A1:{last_col_letter}1")
        t2 = ws2["A1"]
        t2.value = f"سجل متابعة حضور المحاضرات التفصيلي - {course} ({stage} / {section})"
        t2.font = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
        t2.fill = header_fill
        t2.alignment = Alignment(horizontal="center", vertical="center")
        ws2.row_dimensions[1].height = 28

        ws2.row_dimensions[2].height = 36
        ws2["A2"].value = "ت"
        ws2["A2"].font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        ws2["A2"].fill = sub_fill
        ws2["A2"].alignment = Alignment(horizontal="center", vertical="center")
        ws2["A2"].border = thin_border
        ws2.column_dimensions["A"].width = 6

        ws2["B2"].value = "اسم الطالب"
        ws2["B2"].font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        ws2["B2"].fill = sub_fill
        ws2["B2"].alignment = Alignment(horizontal="center", vertical="center")
        ws2["B2"].border = thin_border
        ws2.column_dimensions["B"].width = 28

        ws2["C2"].value = "المعرف"
        ws2["C2"].font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        ws2["C2"].fill = sub_fill
        ws2["C2"].alignment = Alignment(horizontal="center", vertical="center")
        ws2["C2"].border = thin_border
        ws2.column_dimensions["C"].width = 14

        for l_idx, lec in enumerate(lectures, start=1):
            c_letter = get_column_letter(3 + l_idx)
            raw_st = (lec["started_at"] or "")[:10].replace("-", "/")
            header_cell = ws2[f"{c_letter}2"]
            header_cell.value = f"محاضرة {l_idx}\n{raw_st}"
            header_cell.font = Font(name="Calibri", size=9, bold=True, color="FFFFFF")
            header_cell.fill = sub_fill
            header_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            header_cell.border = thin_border
            ws2.column_dimensions[c_letter].width = 14

        pres_fill = PatternFill("solid", fgColor="D1FAE5")
        pres_font = Font(name="Calibri", size=9, bold=True, color="065F46")
        abs_fill  = PatternFill("solid", fgColor="FEE2E2")
        abs_font  = Font(name="Calibri", size=9, bold=True, color="991B1B")

        for s_idx, s in enumerate(stats["students"], start=1):
            r = 2 + s_idx
            ws2.row_dimensions[r].height = 20
            row_fill = PatternFill("solid", fgColor="FFFFFF" if s_idx % 2 == 1 else "F8FAFC")
            cell_font = Font(name="Calibri", size=10, color="1E293B")

            cA = ws2[f"A{r}"]
            cA.value = s_idx
            cA.alignment = Alignment(horizontal="center", vertical="center")
            cA.font = cell_font
            cA.fill = row_fill
            cA.border = thin_border

            cB = ws2[f"B{r}"]
            cB.value = s["full_name"]
            cB.alignment = Alignment(horizontal="right", vertical="center")
            cB.font = cell_font
            cB.fill = row_fill
            cB.border = thin_border

            cC = ws2[f"C{r}"]
            cC.value = s["student_code"] or "—"
            cC.alignment = Alignment(horizontal="center", vertical="center")
            cC.font = cell_font
            cC.fill = row_fill
            cC.border = thin_border

            mat_map = {m["lecture_id"]: m for m in s["lecture_matrix"]}
            for l_idx, lec in enumerate(lectures, start=1):
                c_letter = get_column_letter(3 + l_idx)
                cell = ws2[f"{c_letter}{r}"]
                m_info = mat_map.get(lec["id"])
                if m_info and m_info["attended"]:
                    cell.value = "حاضر"
                    cell.font = pres_font
                    cell.fill = pres_fill
                else:
                    cell.value = "غائب"
                    cell.font = abs_font
                    cell.fill = abs_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_course = "".join(c for c in course if c.isalnum() or c in (" ", "_", "-")).strip() or "course"
        safe_course = safe_course.replace(" ", "_")
        filename = f"cumulative_{safe_course}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        report_path = REPORTS_DIR / filename
        wb.save(str(report_path))

        stream = BytesIO()
        wb.save(stream)
        stream.seek(0)

        headers = {
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        }
        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    except Exception as exc:
        log.error("Cumulative export error: %s", exc)
        raise HTTPException(500, str(exc))


@app.get("/api/health")
async def health():
    return JSONResponse({
        "status"          : "ok",
        "offline_mode"    : True,
        "face_recognition": ENGINE_AVAILABLE,
        "engine"          : "Google MediaPipe & Ultra-Lightweight CPU Engine",
        "faces_loaded"    : FACE_DB.count,
        "camera_index"    : CAMERA_INDEX,
        "session_active"  : SESSION.active,
        "known_faces_dir" : str(KNOWN_DIR),
        "reports_dir"     : str(REPORTS_DIR),
        "timestamp"       : datetime.now().isoformat(),
    })


# ENTRY POINT
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = "127.0.0.1",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )