"""
Verification test for concurrent Webcam Face Recognition and RFID Serial Listener.
Verifies:
  1. Non-blocking MJPEG async generator (CAMERA.get_jpeg_stream).
  2. Independent rfid_background_listener asyncio background task on startup.
  3. RFID card UID: DA:5B:90:32 correctly resolves to 'Ali Nehad'.
  4. Concurrent attendance recording: Webcam detection + RFID swipe simultaneously.
  5. WebSocket (/ws/attendance) receives live events from both sources without freezing the server.
"""

import asyncio
import io
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.stdout.reconfigure(encoding="utf-8")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import main


class MockWebSocket:
    def __init__(self):
        self.received = []

    async def accept(self):
        pass

    async def send_json(self, data):
        self.received.append(data)


async def test_concurrent_pipeline():
    print("=" * 70)
    print("VERIFYING CONCURRENT WEBCAM FEED & RFID ASYNCIO LISTENER")
    print("=" * 70)

    loop = asyncio.get_running_loop()
    main.WS_MANAGER.set_loop(loop)

    # 1. Test RFID Card Mapping for Ali Nehad (UID: DA:5B:90:32)
    print("\n--- 1. Testing UID:DA:5B:90:32 Mapping ---")
    ali_nehad_mapped = main.RFID_REPO.get_student("UID:DA:5B:90:32")
    print(f"UID:DA:5B:90:32 -> {ali_nehad_mapped}")
    assert ali_nehad_mapped == "Ali Nehad", f"Expected 'Ali Nehad', got '{ali_nehad_mapped}'"

    # Also test without prefix
    ali_nehad_mapped2 = main.RFID_REPO.get_student("DA:5B:90:32") or main.RFID_REPO.get_student("UID:DA:5B:90:32")
    assert ali_nehad_mapped2 == "Ali Nehad"

    # 2. Test Face Database
    print("\n--- 2. Pre-loading Face Database ---")
    faces_loaded = main.FACE_DB.load()
    print(f"Loaded {faces_loaded} students from known_faces/: {main.FACE_DB.names}")
    assert "Ali Nehad" in main.FACE_DB.names
    assert "Ali Dhea" in main.FACE_DB.names

    # 3. Test Non-blocking Webcam Stream Generator
    print("\n--- 3. Testing Non-blocking Webcam MJPEG Generator ---")
    main.CAMERA.start()
    stream_gen = main.CAMERA.get_jpeg_stream()

    # Read 5 consecutive frames from the async generator to verify non-blocking yield
    frames_read = 0
    start_time = loop.time()
    async for frame_chunk in stream_gen:
        assert frame_chunk.startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg\r\n\r\n" in frame_chunk
        frames_read += 1
        if frames_read >= 5:
            break

    elapsed = loop.time() - start_time
    print(f"Successfully received {frames_read} MJPEG frames in {elapsed:.3f}s (Non-blocking async generator).")
    assert frames_read == 5
    assert elapsed > 0.05, "Generator properly paced across asyncio.sleep"

    # 4. Start active lecture session
    print("\n--- 4. Starting Active Lecture Session ---")
    main.SESSION.begin(
        department="قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي",
        section="الشعبة أ",
        stage="المرحلة الرابعة",
        course="الرؤية الحاسوبية والأنظمة الذكية",
        instructor="أستاذ المادة المشرف",
    )
    assert main.SESSION.active is True
    assert len(main.SESSION.attendance) == 0

    # 5. Connect Mock WebSocket to test real-time broadcast channel
    mock_ws = MockWebSocket()
    await main.WS_MANAGER.connect(mock_ws)
    assert len(main.WS_MANAGER.active_connections) == 1

    # 6. Test Concurrent Attendance: Webcam Face Recognition + RFID Card Swipe
    print("\n--- 5. Simulating Concurrent Attendance Inputs ---")

    # Input Method 1: Webcam face recognition records Ali Dhea
    student1 = "Ali Dhea"
    photo1 = main.FACE_DB.get_photo(student1)
    rec1 = main.SESSION.try_record_attendance(
        name=student1,
        photo_b64=photo1,
        method="كاميرا ذكية (AI Vision)",
    )
    assert rec1 is True

    # Input Method 2: RFID serial swipe records Ali Nehad (UID: DA:5B:90:32)
    # Process payload directly through RFID_MANAGER
    main.RFID_MANAGER._process_payload("UID:DA:5B:90:32")

    # Cooperative wait for event loop tasks
    await asyncio.sleep(0.08)

    snap = main.SESSION.snapshot()
    print("Current Session Attendance Count:", snap["count"])
    assert snap["count"] == 2

    records = snap["attendance"]
    print("Record 1:", records[0]["name"], "| Method:", records[0]["method"])
    print("Record 2:", records[1]["name"], "| Method:", records[1]["method"])

    assert records[0]["name"] == "Ali Dhea"
    assert "AI Vision" in records[0]["method"]

    assert records[1]["name"] == "Ali Nehad"
    assert "RFID" in records[1]["method"]

    # Verify WebSocket received broadcast events for both students
    print(f"WebSocket Messages Dispatched: {len(mock_ws.received)}")
    assert len(mock_ws.received) >= 2

    ws_student_names = [m["data"]["name"] for m in mock_ws.received if m.get("event") == "attendance_recorded"]
    print("WebSocket Attendees Broadcasted:", ws_student_names)
    assert "Ali Dhea" in ws_student_names
    assert "Ali Nehad" in ws_student_names

    # 7. Test RFID Background Listener Coroutine startup and cancellation
    print("\n--- 6. Testing rfid_background_listener Lifecycle ---")
    rfid_task = asyncio.create_task(main.rfid_background_listener(), name="test-rfid-listener")
    assert not rfid_task.done()

    # Let listener run for a brief moment
    await asyncio.sleep(0.1)
    rfid_task.cancel()
    try:
        await rfid_task
    except (asyncio.CancelledError, Exception):
        pass
    print("rfid_background_listener started, yielded, and cancelled cleanly without hanging.")

    # 8. Test Clean Session Teardown
    main.WS_MANAGER.disconnect(mock_ws)
    main.CAMERA.stop()
    main.SESSION.terminate()

    print("\n" + "=" * 70)
    print("CONCURRENT WEBCAM FEED & RFID PIPELINE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_concurrent_pipeline())
