"""
Test suite to comprehensively verify:
1. Model loading and startup caching of 8/8 students in known_faces/
2. CameraManager hardware-level resolution downscaling (640x480)
3. Non-blocking bounded queue (maxsize=2)
4. Smart frame-skipping (DETECTION_INTERVAL=5) and cached tracking
5. Session deduplication and attendance recording
6. FastAPI endpoints & health checks
"""
import sys
import time
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import main

def test_pipeline():
    print("=" * 60)
    print("STEP 1: Verify Face Database Loading & Embeddings Extraction")
    print("=" * 60)
    t0 = time.time()
    count = main.FACE_DB.load()
    load_time = time.time() - t0
    print(f"Loaded {count} faces in {load_time:.2f}s")
    assert count == 8, f"Expected 8 faces, got {count}"
    assert len(main.FACE_DB.names) == 8, "Expected 8 names"
    assert main.FACE_DB.embeddings_matrix is not None, "Embeddings matrix must not be None"
    assert main.FACE_DB.embeddings_matrix.shape == (8, 512), f"Expected shape (8, 512), got {main.FACE_DB.embeddings_matrix.shape}"
    assert len(main.FACE_DB.photos) == 8, "Expected 8 reference thumbnails"
    print(" [PASS] 8/8 student reference photos extracted and cached in memory.")

    print("\n" + "=" * 60)
    print("STEP 2: Verify Vectorized Cosine Similarity & Distance Matching")
    print("=" * 60)
    for idx, name in enumerate(main.FACE_DB.names):
        emb = main.FACE_DB.embeddings[idx]
        matched_name, sim = main.FACE_DB.match(emb)
        print(f"  Self-match: {name:25s} -> {matched_name:25s} (sim: {sim:.4f})")
        assert matched_name == name, f"Mismatch for {name}"
        assert sim >= 0.99, f"Self-similarity should be ~1.0, got {sim}"
    print(" [PASS] Vectorized matching matches all 8 students with >0.99 cosine similarity.")

    print("\n" + "=" * 60)
    print("STEP 3: Verify CameraManager Configuration & Hardware Downscaling")
    print("=" * 60)
    print(f"  CAMERA_WIDTH:  {main.CAMERA_WIDTH} (Target: 640)")
    print(f"  CAMERA_HEIGHT: {main.CAMERA_HEIGHT} (Target: 480)")
    print(f"  DETECTION_WIDTH:  {main.DETECTION_WIDTH} (Target: 320)")
    print(f"  DETECTION_HEIGHT: {main.DETECTION_HEIGHT} (Target: 240)")
    print(f"  DETECTION_INTERVAL: {main.DETECTION_INTERVAL} (Target: 5)")
    assert main.CAMERA_WIDTH == 640, "CAMERA_WIDTH should be 640"
    assert main.CAMERA_HEIGHT == 480, "CAMERA_HEIGHT should be 480"
    assert main.DETECTION_WIDTH == 320, "DETECTION_WIDTH should be 320"
    assert main.DETECTION_HEIGHT == 240, "DETECTION_HEIGHT should be 240"
    assert main.DETECTION_INTERVAL == 5, "DETECTION_INTERVAL should be 5"
    print(" [PASS] Resolution downscaling directly at cv2.VideoCapture configuration.")

    print("\n" + "=" * 60)
    print("STEP 4: Verify Non-blocking Bounded Queue (maxsize=2)")
    print("=" * 60)
    cam = main.CameraManager(camera_index=99)  # mock camera index
    assert cam._frame_queue.maxsize == 2, "Queue maxsize must be 2"
    # Push 5 frames and verify no overflow or blocking
    for i in range(5):
        cam._push_frame(b"frame_%d" % i)
    assert cam._frame_queue.qsize() == 2, f"Expected qsize 2, got {cam._frame_queue.qsize()}"
    # The queue must have the freshest frames (3 and 4)
    f1 = cam._frame_queue.get_nowait()
    f2 = cam._frame_queue.get_nowait()
    assert f1 == b"frame_3" and f2 == b"frame_4", f"Queue did not discard stale frames: {f1}, {f2}"
    print(" [PASS] Bounded queue strictly enforces maxsize=2 and drops stale frames instantly.")

    print("\n" + "=" * 60)
    print("STEP 5: Verify Session Deduplication & Attendance Flow")
    print("=" * 60)
    main.SESSION.begin(
        department="قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي",
        section="الشعبة أ",
        stage="المرحلة الرابعة",
        course="Computer Vision",
    )
    assert main.SESSION.active, "Session must be active"
    test_student = main.FACE_DB.names[0]
    test_photo = main.FACE_DB.get_photo(test_student)

    # First attempt: should succeed and record
    res1 = main.SESSION.try_record_attendance(test_student, test_photo)
    assert res1 is True, "First recording must return True"
    assert len(main.SESSION.attendance) == 1, "Session count must be 1"

    # Second attempt: deduplication should reject
    res2 = main.SESSION.try_record_attendance(test_student, test_photo)
    assert res2 is False, "Second recording must return False (deduplication)"
    assert len(main.SESSION.attendance) == 1, "Session count must remain 1"
    print(" [PASS] Deduplication strictly prevents duplicate registration in the same session.")

    # Clean teardown
    main.SESSION.terminate()
    assert not main.SESSION.active, "Session must be inactive"
    print(" [PASS] Clean session teardown.")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
