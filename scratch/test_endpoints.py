"""
Comprehensive FastAPI endpoint testing script.
Directly tests async handlers using exact route function names.
"""
import asyncio
import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.stdout.reconfigure(encoding="utf-8")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import main

async def test_all_endpoints():
    print("=" * 60)
    print("TESTING FASTAPI ENDPOINTS & ASYNC LOGIC")
    print("=" * 60)

    # 1. Startup Lifespan / Pre-load
    count = main.FACE_DB.load()
    print(f"Face DB loaded: {count} student reference profiles.")
    assert count == 8, f"Expected 8 profiles, got {count}"

    # 2. Health Endpoint
    health_resp = await main.health()
    health_data = json.loads(health_resp.body.decode())
    print("Health response:", health_data)
    assert health_data["status"] == "ok"
    assert health_data["face_recognition"] is True
    assert health_data["faces_loaded"] == 8

    # 3. Face DB Status Endpoint
    db_resp = await main.face_db_status()
    db_data = json.loads(db_resp.body.decode())
    print("Face DB Status count:", db_data["count"], "names:", db_data["names"])
    assert db_data["count"] == 8
    assert len(db_data["names"]) == 8

    # 4. Live Session State (inactive)
    state_resp = await main.session_state()
    state_data = json.loads(state_resp.body.decode())
    print("Session State (inactive): active =", state_data["active"])
    assert state_data["active"] is False
    assert state_data["count"] == 0

    # 5. Start Session Form Submit
    start_resp = await main.start_session(
        department="قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي",
        section="الشعبة أ",
        stage="المرحلة الرابعة",
        course="Computer Vision & Deep Learning",
    )
    print("Start session redirect status:", start_resp.status_code, "headers:", start_resp.headers.get("location"))
    assert start_resp.status_code == 303
    assert start_resp.headers.get("location") == "/lecture"
    assert main.SESSION.active is True
    assert main.SESSION.course == "Computer Vision & Deep Learning"

    # 6. Session State (active)
    state_active_resp = await main.session_state()
    state_active = json.loads(state_active_resp.body.decode())
    print("Session State (active):", state_active["department"], state_active["course"])
    assert state_active["active"] is True
    assert state_active["department"] == "قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي"

    # 7. Record an attendance entry
    student_name = main.FACE_DB.names[0]
    student_photo = main.FACE_DB.get_photo(student_name)
    rec1 = main.SESSION.try_record_attendance(student_name, student_photo)
    assert rec1 is True
    assert len(main.SESSION.attendance) == 1

    # 8. Session State with 1 attendee
    state_with_student_resp = await main.session_state()
    state_with_student = json.loads(state_with_student_resp.body.decode())
    print("Session State with 1 student count:", state_with_student["count"], "student:", state_with_student["attendance"][0]["name"])
    assert state_with_student["count"] == 1
    assert state_with_student["attendance"][0]["name"] == student_name

    # 9. End Session
    end_resp = await main.end_session()
    end_data = json.loads(end_resp.body.decode())
    print("End session response:", end_data)
    assert end_data["ok"] is True
    assert main.SESSION.active is False

    # 10. Verify Setup Page rendering
    setup_page_resp = await main.setup_dashboard()
    print("Setup page response status:", setup_page_resp.status_code)
    assert setup_page_resp.status_code == 200

    print("\n" + "=" * 60)
    print("ALL FASTAPI ASYNC ENDPOINTS VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all_endpoints())
