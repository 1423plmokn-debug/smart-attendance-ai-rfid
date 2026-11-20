"""
Comprehensive automated integration test for ESP32-S + RC522 RFID secondary attendance pipeline.
Tests:
  1. RFIDCardRepository: default mappings, dynamic resolution, auto-assignment for unmapped cards.
  2. RFIDSerialManager payload processing: regex parsing of "UID:XX:XX:XX:XX", debouncing.
  3. Session Attendance Logging: recorded with method="بطاقة جامعية (RFID RC522)".
  4. Real-time WebSocket (/ws/attendance): client connection, live broadcast event dispatch.
  5. REST Endpoints: /api/rfid-status and /api/rfid-map.
  6. Integration with Excel export (/api/export-report) verifying RFID records are preserved.
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

import openpyxl
import main


class MockWebSocket:
    """Mock WebSocket client for testing /ws/attendance broadcast pipeline."""
    def __init__(self):
        self.received = []

    async def accept(self):
        pass

    async def send_json(self, data):
        self.received.append(data)


async def run_rfid_integration_tests():
    print("=" * 60)
    print("TESTING ESP32-S + RC522 RFID ATTENDANCE PIPELINE")
    print("=" * 60)

    # Register active running asyncio loop with WebSocket manager
    loop = asyncio.get_running_loop()
    main.WS_MANAGER.set_loop(loop)

    # 1. Test RFID Card Repository
    print("\n--- 1. Testing RFIDCardRepository ---")
    repo = main.RFID_REPO
    print("Loaded mappings count:", len(repo.mappings))
    assert len(repo.mappings) >= 8, "Expected at least 8 student mappings"

    ali_student = repo.get_student("UID:3A:5B:92:F1")
    print("Mapped UID:3A:5B:92:F1 ->", ali_student)
    assert ali_student == "Ali Dhea", f"Expected Ali Dhea, got {ali_student}"

    # Test auto-assignment of an unknown card UID
    unmapped_uid = "UID:FE:DC:BA:98"
    assigned = repo.resolve_or_auto_assign(unmapped_uid, main.FACE_DB.names or ["Mahdi Ali"])
    print(f"Auto-assigned unmapped {unmapped_uid} ->", assigned)
    assert assigned is not None
    assert repo.get_student(unmapped_uid) == assigned

    # 2. Test REST Endpoints
    print("\n--- 2. Testing REST Endpoints ---")
    # Test /api/rfid-status
    resp_status = await main.rfid_status()
    status_data = json.loads(resp_status.body.decode("utf-8"))
    print("RFID Status Response:", status_data)
    assert status_data["ok"] is True
    assert status_data["port"] == "COM5"
    assert status_data["baudrate"] == 115200
    assert status_data["serial_available"] is True

    # Test /api/rfid-map
    resp_map = await main.rfid_map(uid="99:88:77:66", student_name="Test Student")
    map_data = json.loads(resp_map.body.decode("utf-8"))
    print("RFID Map Response:", map_data)
    assert map_data["ok"] is True
    assert map_data["uid"] == "UID:99:88:77:66"
    assert repo.get_student("UID:99:88:77:66") == "Test Student"

    # 3. Test Active Session Attendance Logging via RFID Serial Manager
    print("\n--- 3. Testing Session Attendance via RFID ---")
    main.FACE_DB.load()

    # Start a mock active lecture session
    main.SESSION.begin(
        department="قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي",
        section="الشعبة أ",
        stage="المرحلة الرابعة",
        course="الأنظمة المضمنة الذكية (Smart Embedded Systems)",
        instructor="أستاذ المادة المشرف",
    )

    # 4. Connect Mock WebSocket to test real-time broadcast
    mock_ws = MockWebSocket()
    await main.WS_MANAGER.connect(mock_ws)
    assert len(main.WS_MANAGER.active_connections) == 1

    # Simulate ESP32 serial card swipe payload: "UID:3A:5B:92:F1"
    main.RFID_MANAGER._process_payload("UID:3A:5B:92:F1")
    await asyncio.sleep(0.05)  # Cooperative sleep for thread-safe broadcast execution

    snap = main.SESSION.snapshot()
    print("Session Attendance Count after 1st RFID swipe:", snap["count"])
    assert snap["count"] == 1
    first_record = snap["attendance"][0]
    print("First record:", first_record)
    assert first_record["name"] == "Ali Dhea"
    assert first_record["method"] == "بطاقة جامعية (RFID RC522)"

    # Verify WebSocket received the broadcast
    print("WebSocket Messages Received:", len(mock_ws.received))
    assert len(mock_ws.received) >= 1
    last_ws_msg = mock_ws.received[-1]
    print("WebSocket Last Message:", last_ws_msg)
    assert last_ws_msg["event"] == "attendance_recorded"
    assert last_ws_msg["data"]["name"] == "Ali Dhea"
    assert last_ws_msg["data"]["method"] == "بطاقة جامعية (RFID RC522)"

    # Test Debounce Cooldown: Immediate second swipe of identical card must be ignored
    main.RFID_MANAGER._process_payload("UID:3A:5B:92:F1")
    snap2 = main.SESSION.snapshot()
    print("Session Attendance Count after duplicate swipe:", snap2["count"])
    assert snap2["count"] == 1, "Duplicate swipe within cooldown window must not be recorded again"

    # Swipe second student card: "UID:4C:6D:81:A2" (Ali Nehad)
    main.RFID_MANAGER._process_payload("UID:4C:6D:81:A2")
    await asyncio.sleep(0.05)

    snap3 = main.SESSION.snapshot()
    print("Session Attendance Count after 2nd student RFID swipe:", snap3["count"])
    assert snap3["count"] == 2
    assert snap3["attendance"][1]["name"] == "Ali Nehad"
    assert snap3["attendance"][1]["method"] == "بطاقة جامعية (RFID RC522)"

    # Swipe third student: "UID:B3:99:A1:02" (Baneen Talib)
    main.RFID_MANAGER._process_payload("UID:B3:99:A1:02")
    await asyncio.sleep(0.05)

    snap4 = main.SESSION.snapshot()
    assert snap4["count"] == 3
    print("Session Attendance Count after 3rd student RFID swipe:", snap4["count"])

    # Disconnect Mock WebSocket
    main.WS_MANAGER.disconnect(mock_ws)
    assert len(main.WS_MANAGER.active_connections) == 0

    # 5. Test Excel Export includes RFID registered students
    print("\n--- 5. Testing Excel Export with RFID records ---")
    resp_export = await main.export_report()
    body = b""
    async for chunk in resp_export.body_iterator:
        body += chunk

    wb = openpyxl.load_workbook(io.BytesIO(body))
    ws = wb.active

    # Check rows for RFID registration method in column D
    rfid_rows = []
    for r in range(9, 17):
        name = ws.cell(row=r, column=2).value
        method = ws.cell(row=r, column=4).value
        status = ws.cell(row=r, column=6).value
        if "RFID" in str(method):
            rfid_rows.append((name, method, status))
            print(f"  Excel Row {r}: {name:15s} | {method:28s} | {status}")

    print("Total RFID verified attendees in Excel report:", len(rfid_rows))
    assert len(rfid_rows) == 3, f"Expected 3 RFID attendees in report, got {len(rfid_rows)}"

    main.SESSION.terminate()
    print("\n" + "=" * 60)
    print("ALL RFID INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_rfid_integration_tests())
