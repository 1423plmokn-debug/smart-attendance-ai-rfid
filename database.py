# Copyright (c) 2026 Ali Nihad Qadir -- Middle Technical University / Polytechnic College.
# All Rights Reserved. Unauthorized copying, redistribution, or attribution is prohibited.
"""
database.py - Offline SQLite Repository for Smart Attendance System.
Zero external dependencies, thread-safe with threading.Lock and check_same_thread=False.
"""

from __future__ import annotations

import sqlite3
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("attendance.database")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "attendance.db"

def embedding_to_blob(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()

def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name    TEXT    NOT NULL,
    student_code TEXT,
    department   TEXT    NOT NULL,
    stage        TEXT    NOT NULL,
    section      TEXT    NOT NULL,
    rfid_uid     TEXT    UNIQUE,
    photo_path   TEXT,
    embedding    BLOB    NOT NULL,
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS lectures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    department  TEXT    NOT NULL,
    stage       TEXT    NOT NULL,
    section     TEXT    NOT NULL,
    course      TEXT    NOT NULL,
    instructor  TEXT,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT
);

CREATE TABLE IF NOT EXISTS attendance_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lecture_id  INTEGER NOT NULL REFERENCES lectures(id)  ON DELETE CASCADE,
    student_id  INTEGER NOT NULL REFERENCES students(id)  ON DELETE CASCADE,
    timestamp   TEXT    NOT NULL,
    method      TEXT    NOT NULL,
    UNIQUE(lecture_id, student_id)
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS academic_items (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT    NOT NULL,
    name     TEXT    NOT NULL,
    UNIQUE(category, name)
);
"""

DEFAULT_INSTITUTION_AR = "الجامعة التقنية الوسطى / كلية البوليتكنك"
DEFAULT_INSTITUTION_EN = "Middle Technical University - Polytechnic College"

DEFAULT_DEPARTMENTS: list[str] = [
    "قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي",
    "قسم هندسة التقنيات الكيماوية والصناعات النفطية",
    "قسم هندسة تقنيات التصنيع المبرمج",
    "قسم هندسة تقنيات الكهربائية والأنظمة الذكية",
    "قسم هندسة التقنيات المدنية وإدارة المشاريع",
    "قسم هندسة تقنيات ميكانيك القدرة والطاقة المتجددة",
    "قسم هندسة تقنيات مشاريع المياه والبيئة",
    "قسم هندسة تكنولوجيا المعلومات والأمن السيبراني",
    "قسم تقنيات المساحة",
    "قسم تقنيات الأجهزة الطبية",
]
DEFAULT_SECTIONS: list[str] = ["الشعبة أ", "الشعبة ب", "الشعبة ج", "الشعبة د", "الشعبة هـ"]
DEFAULT_STAGES: list[str] = ["المرحلة الأولى", "المرحلة الثانية", "المرحلة الثالثة", "المرحلة الرابعة"]

class AttendanceDB:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                timeout=15.0,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
        return self._conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return dict(row)

    def init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

            cur = conn.execute("SELECT COUNT(*) as cnt FROM settings")
            if cur.fetchone()["cnt"] == 0:
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?)",
                    ("institution_ar", DEFAULT_INSTITUTION_AR),
                )
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?)",
                    ("institution_en", DEFAULT_INSTITUTION_EN),
                )
                conn.commit()

            def _seed_category(cat: str, defaults: list[str]):
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM academic_items WHERE category = ?", (cat,)
                ).fetchone()
                if row["cnt"] == 0:
                    for name in defaults:
                        conn.execute(
                            "INSERT OR IGNORE INTO academic_items (category, name) VALUES (?, ?)",
                            (cat, name),
                        )
                    conn.commit()

            _seed_category("department", DEFAULT_DEPARTMENTS)
            _seed_category("stage", DEFAULT_STAGES)
            _seed_category("section", DEFAULT_SECTIONS)

        log.info("SQLite database initialised at %s", self._path)

    def add_student(
        self,
        full_name: str,
        department: str,
        stage: str,
        section: str,
        embedding: np.ndarray,
        photo_path: Optional[str] = None,
        student_code: Optional[str] = None,
        rfid_uid: Optional[str] = None,
    ) -> int:
        blob = embedding_to_blob(embedding)
        created_at = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                """
                INSERT INTO students
                    (full_name, student_code, department, stage, section,
                     rfid_uid, photo_path, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (full_name, student_code, department, stage, section,
                 rfid_uid, photo_path, blob, created_at),
            )
            conn.commit()
            new_id = cur.lastrowid
        log.info("Student added: id=%d name=%s", new_id, full_name)
        return new_id

    def get_students(
        self,
        department: Optional[str] = None,
        stage: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[str] = []
        if department:
            clauses.append("department = ?")
            params.append(department)
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if section:
            clauses.append("section = ?")
            params.append(section)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT id, full_name, student_code, department, stage, section,
                   rfid_uid, photo_path, created_at
            FROM students
            {where}
            ORDER BY full_name
        """
        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_students_with_embeddings(
        self,
        department: Optional[str] = None,
        stage: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[str] = []
        if department:
            clauses.append("department = ?")
            params.append(department)
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if section:
            clauses.append("section = ?")
            params.append(section)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT id, full_name, student_code, department, stage, section,
                   rfid_uid, photo_path, embedding, created_at
            FROM students
            {where}
            ORDER BY full_name
        """
        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_student_by_rfid(self, rfid_uid: str) -> Optional[dict]:
        if not rfid_uid:
            return None
        clean = rfid_uid.strip().upper()
        clean_prefix = clean if clean.startswith("UID:") else f"UID:{clean}"
        clean_no_prefix = clean[4:] if clean.startswith("UID:") else clean
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM students WHERE rfid_uid = ? OR rfid_uid = ? OR rfid_uid = ?",
                (rfid_uid, clean_prefix, clean_no_prefix),
            ).fetchone()
        if row is None:
            return None
        d = self._row_to_dict(row)
        if d.get("embedding"):
            d["embedding"] = blob_to_embedding(d["embedding"])
        return d

    def delete_student(self, student_id: int) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
            conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            log.info("Student deleted: id=%d", student_id)
        return deleted

    def create_lecture(
        self,
        department: str,
        stage: str,
        section: str,
        course: str,
        instructor: Optional[str] = None,
    ) -> int:
        started_at = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                """
                INSERT INTO lectures
                    (department, stage, section, course, instructor, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (department, stage, section, course, instructor, started_at),
            )
            conn.commit()
            lecture_id = cur.lastrowid
        log.info("Lecture created: id=%d course=%s sec=%s", lecture_id, course, section)
        return lecture_id

    def end_lecture(self, lecture_id: int) -> None:
        ended_at = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE lectures SET ended_at = ? WHERE id = ?",
                (ended_at, lecture_id),
            )
            conn.commit()
        log.info("Lecture ended: id=%d", lecture_id)

    def get_active_lecture(self) -> Optional[dict]:
        with self._lock:
            row = self._get_conn().execute(
                """
                SELECT * FROM lectures
                WHERE ended_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def record_attendance(
        self,
        lecture_id: int,
        student_id: int,
        method: str,
    ) -> bool:
        timestamp = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO attendance_records
                        (lecture_id, student_id, timestamp, method)
                    VALUES (?, ?, ?, ?)
                    """,
                    (lecture_id, student_id, timestamp, method),
                )
                conn.commit()
                log.info(
                    "Attendance recorded: lecture=%d student=%d method=%s",
                    lecture_id, student_id, method,
                )
                return True
            except sqlite3.IntegrityError:
                log.debug(
                    "Duplicate attendance ignored: lecture=%d student=%d",
                    lecture_id, student_id,
                )
                return False

    def get_lecture_attendance(self, lecture_id: int) -> list[dict]:
        sql = """
            SELECT
                ar.id,
                ar.lecture_id,
                ar.student_id,
                ar.timestamp,
                ar.method,
                s.full_name,
                s.student_code,
                s.department,
                s.stage,
                s.section,
                s.rfid_uid,
                s.photo_path
            FROM attendance_records ar
            JOIN students s ON s.id = ar.student_id
            WHERE ar.lecture_id = ?
            ORDER BY ar.timestamp ASC
        """
        with self._lock:
            rows = self._get_conn().execute(sql, (lecture_id,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_cumulative_stats(
        self,
        department: str,
        stage: str,
        section: str,
        course: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> dict:
        lec_sql = """
            SELECT id, department, stage, section, course, instructor, started_at, ended_at
            FROM lectures
            WHERE department = ? AND stage = ? AND section = ? AND course = ?
        """
        lec_params: list = [department, stage, section, course]

        if date_from and date_from.strip():
            df = date_from.strip()
            df_iso = df if "T" in df else f"{df}T00:00:00"
            lec_sql += " AND started_at >= ?"
            lec_params.append(df_iso)

        if date_to and date_to.strip():
            dt = date_to.strip()
            dt_iso = dt if "T" in dt else f"{dt}T23:59:59.999"
            lec_sql += " AND started_at <= ?"
            lec_params.append(dt_iso)

        lec_sql += " ORDER BY started_at ASC"

        with self._lock:
            conn = self._get_conn()
            lecture_rows = conn.execute(lec_sql, lec_params).fetchall()
            lectures = [self._row_to_dict(r) for r in lecture_rows]

            st_sql = """
                SELECT id, full_name, student_code, department, stage, section, rfid_uid, photo_path
                FROM students
                WHERE department = ? AND stage = ? AND section = ?
                ORDER BY full_name ASC
            """
            student_rows = conn.execute(st_sql, (department, stage, section)).fetchall()
            students = [self._row_to_dict(r) for r in student_rows]

            total_lectures = len(lectures)
            lecture_ids = [l["id"] for l in lectures]

            records_by_student: dict[int, dict[int, dict]] = {s["id"]: {} for s in students}

            if lecture_ids:
                placeholders = ",".join("?" * len(lecture_ids))
                rec_sql = f"""
                    SELECT lecture_id, student_id, timestamp, method
                    FROM attendance_records
                    WHERE lecture_id IN ({placeholders})
                """
                rec_rows = conn.execute(rec_sql, lecture_ids).fetchall()
                for r in rec_rows:
                    sid = r["student_id"]
                    lid = r["lecture_id"]
                    if sid in records_by_student:
                        records_by_student[sid][lid] = self._row_to_dict(r)

            student_stats = []
            for s in students:
                sid = s["id"]
                attended_map = records_by_student.get(sid, {})
                present_count = len(attended_map)
                absent_count = max(0, total_lectures - present_count)
                att_pct = round(present_count / total_lectures * 100, 1) if total_lectures > 0 else 0.0
                abs_pct = round(absent_count / total_lectures * 100, 1) if total_lectures > 0 else 0.0

                if abs_pct >= 15.0:
                    status = "محروم"
                elif abs_pct >= 10.0:
                    status = "إنذار أول"
                else:
                    status = "منتظم"

                lecture_matrix = []
                for l in lectures:
                    lid = l["id"]
                    att_rec = attended_map.get(lid)
                    lecture_matrix.append({
                        "lecture_id": lid,
                        "started_at": l["started_at"],
                        "attended": att_rec is not None,
                        "method": att_rec["method"] if att_rec else "",
                        "timestamp": att_rec["timestamp"] if att_rec else "",
                    })

                student_stats.append({
                    "student_id": sid,
                    "full_name": s["full_name"],
                    "student_code": s.get("student_code") or "",
                    "rfid_uid": s.get("rfid_uid") or "",
                    "total_lectures": total_lectures,
                    "present_count": present_count,
                    "absent_count": absent_count,
                    "attendance_pct": att_pct,
                    "absence_pct": abs_pct,
                    "status": status,
                    "lecture_matrix": lecture_matrix,
                })

            avg_attendance_pct = (
                round(sum(st["attendance_pct"] for st in student_stats) / len(student_stats), 1)
                if student_stats else 0.0
            )

            deprived_count = sum(1 for st in student_stats if st["status"] == "محروم")
            warning_count = sum(1 for st in student_stats if st["status"] == "إنذار أول")

            return {
                "department": department,
                "stage": stage,
                "section": section,
                "course": course,
                "total_lectures": total_lectures,
                "total_students": len(students),
                "avg_attendance_pct": avg_attendance_pct,
                "deprived_count": deprived_count,
                "warning_count": warning_count,
                "lectures": [
                    {
                        "id": l["id"],
                        "started_at": l["started_at"],
                        "ended_at": l.get("ended_at"),
                        "instructor": l.get("instructor") or "",
                    }
                    for l in lectures
                ],
                "students": student_stats,
            }

    def get_setting(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value.strip()),
            )
            conn.commit()

    def get_academic_items(self, category: str) -> list[dict]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT id, category, name FROM academic_items WHERE category = ? ORDER BY id",
                (category,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def add_academic_item(self, category: str, name: str) -> int:
        clean_cat = category.strip().lower()
        clean_name = name.strip()
        if clean_cat not in ("department", "stage", "section"):
            raise ValueError(f"Invalid category: {category}")
        if not clean_name:
            raise ValueError("Item name cannot be empty")

        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "INSERT INTO academic_items (category, name) VALUES (?, ?)",
                    (clean_cat, clean_name),
                )
                conn.commit()
                new_id = cur.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError(f"العنصر '{clean_name}' موجود بالفعل في هذا التصنيف")
        log.info("Academic item added: category=%s name=%s (id=%d)", clean_cat, clean_name, new_id)
        return new_id

    def rename_academic_item(self, item_id: int, new_name: str) -> bool:
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("New name cannot be empty")

        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT category, name FROM academic_items WHERE id = ?", (item_id,)).fetchone()
            if not row:
                return False
            category = row["category"]
            old_name = row["name"]

            if old_name == clean_name:
                return True

            try:
                conn.execute("UPDATE academic_items SET name = ? WHERE id = ?", (clean_name, item_id))
                if category == "department":
                    conn.execute("UPDATE students SET department = ? WHERE department = ?", (clean_name, old_name))
                    conn.execute("UPDATE lectures SET department = ? WHERE department = ?", (clean_name, old_name))
                elif category == "stage":
                    conn.execute("UPDATE students SET stage = ? WHERE stage = ?", (clean_name, old_name))
                    conn.execute("UPDATE lectures SET stage = ? WHERE stage = ?", (clean_name, old_name))
                elif category == "section":
                    conn.execute("UPDATE students SET section = ? WHERE section = ?", (clean_name, old_name))
                    conn.execute("UPDATE lectures SET section = ? WHERE section = ?", (clean_name, old_name))
                conn.commit()
                log.info("Renamed academic item %d (%s) from '%s' to '%s' with cascade.", item_id, category, old_name, clean_name)
                return True
            except sqlite3.IntegrityError:
                conn.rollback()
                raise ValueError(f"الاسم '{clean_name}' موجود مسبقاً في هذا التصنيف")

    def delete_academic_item(self, item_id: int) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM academic_items WHERE id = ?", (item_id,))
            conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            log.info("Academic item deleted: id=%d", item_id)
        return deleted

    def update_student_rfid(self, student_id: int, rfid_uid: Optional[str]) -> bool:
        clean_uid = rfid_uid.strip().upper() if rfid_uid and rfid_uid.strip() else None
        if clean_uid and not clean_uid.startswith("UID:"):
            clean_uid = f"UID:{clean_uid}"

        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute("UPDATE students SET rfid_uid = ? WHERE id = ?", (clean_uid, student_id))
                conn.commit()
                return cur.rowcount > 0
            except sqlite3.IntegrityError:
                raise ValueError(f"البطاقة الجامعية '{clean_uid}' مسجلة بالفعل لطالب آخر")

    def get_distinct_courses(
        self,
        department: Optional[str] = None,
        stage: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[str]:
        clauses = []
        params = []
        if department:
            clauses.append("department = ?")
            params.append(department)
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if section:
            clauses.append("section = ?")
            params.append(section)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT DISTINCT course FROM lectures {where} ORDER BY course ASC"
        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()
        return [r["course"] for r in rows if r["course"]]

DB = AttendanceDB()
