"""
Comprehensive test for the dynamic institutional /api/export-report endpoint.
"""
import asyncio
import io
import os
import sys
from pathlib import Path
import openpyxl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.stdout.reconfigure(encoding="utf-8")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import main

async def test_export_endpoint():
    print("=" * 60)
    print("TESTING DYNAMIC INSTITUTIONAL /api/export-report ENDPOINT")
    print("=" * 60)

    # Pre-load face database so 8 students are known
    count = main.FACE_DB.load()
    print(f"Face DB loaded: {count} reference students.")

    # Start a mock active session
    main.SESSION.begin(
        department="قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي",
        section="الشعبة ب",
        stage="المرحلة الرابعة",
        course="نظم الرؤية الحاسوبية (Computer Vision Systems)",
        instructor="أ.م.د. علي التميمي",
    )

    # Record 3 students as present
    p1 = main.FACE_DB.names[0]
    p2 = main.FACE_DB.names[1]
    p3 = main.FACE_DB.names[2]
    main.SESSION.try_record_attendance(p1, main.FACE_DB.get_photo(p1), method="كاميرا ذكية (AI Vision)")
    main.SESSION.try_record_attendance(p2, main.FACE_DB.get_photo(p2), method="كاميرا ذكية (AI Vision)")
    main.SESSION.try_record_attendance(p3, main.FACE_DB.get_photo(p3), method="بصمة الوجه (Face Biometrics)")

    print(f"Recorded 3 attendees: {p1}, {p2}, {p3}")

    # TEST 1: Export with fallback to session
    print("\n--- Test 1: Fallback to Active Session ---")
    resp1 = await main.export_report()
    body1 = b""
    async for chunk in resp1.body_iterator:
        body1 += chunk

    wb1 = openpyxl.load_workbook(io.BytesIO(body1))
    ws1 = wb1.active
    print("Sheet Title:", ws1.title)
    print("RTL enabled:", ws1.views.sheetView[0].rightToLeft)
    assert ws1.views.sheetView[0].rightToLeft is True, "Sheet must be RTL"

    # Header check
    assert ws1["A2"].value == "الجامعة التقنية الوسطى / كلية البوليتكنك"
    assert "قسم هندسة تقنيات الإلكترونيك والذكاء الاصطناعي" in ws1["A3"].value
    print("Header A2:", ws1["A2"].value)
    print("Subheader A3:", ws1["A3"].value)

    # Metadata check
    print("Subject (B5):", ws1["B5"].value)
    print("Section (F5):", ws1["F5"].value)
    print("Supervising Prof Label (A6):", ws1["A6"].value)
    print("Supervising Prof Value (B6):", repr(ws1["B6"].value))
    assert ws1["B5"].value == "نظم الرؤية الحاسوبية (Computer Vision Systems)"
    assert ws1["F5"].value == "الشعبة ب"
    assert ws1["A6"].value == "أستاذ المادة المشرف:"
    assert ws1["B6"].value in (None, "")

    # Data rows check
    # Row 8 is header: ت, اسم الطالب, القسم / الشعبة, طريقة التسجيل, وقت الحضور, حالة الحضور
    headers = [ws1.cell(row=8, column=col).value for col in range(1, 7)]
    print("Table Headers (Row 8):", headers)
    assert headers == ["ت", "اسم الطالب", "القسم / الشعبة", "طريقة التسجيل", "وقت الحضور", "حالة الحضور"]

    # Rows 9 to 16 (8 students total)
    student_records = []
    for r in range(9, 17):
        idx = ws1.cell(row=r, column=1).value
        name = ws1.cell(row=r, column=2).value
        dept_sec = ws1.cell(row=r, column=3).value
        meth = ws1.cell(row=r, column=4).value
        ts = ws1.cell(row=r, column=5).value
        stat = ws1.cell(row=r, column=6).value
        student_records.append((idx, name, stat, meth))
        print(f"  Row {r}: [{idx}] {name:20s} | {stat:10s} | {meth}")

    assert len(student_records) == 8, f"Expected 8 rows, got {len(student_records)}"
    p_count = sum(1 for _, _, stat, _ in student_records if "حاضر" in stat)
    a_count = sum(1 for _, _, stat, _ in student_records if "غائب" in stat)
    print(f"Present count: {p_count}, Absent count: {a_count}")
    assert p_count == 3, f"Expected 3 present, got {p_count}"
    assert a_count == 5, f"Expected 5 absent, got {a_count}"

    # Summary metrics check
    # Row 17 is space, row 18 is title, row 19 is values
    print("Summary Total (A19):", ws1["A19"].value)
    print("Summary Present (C19):", ws1["C19"].value)
    print("Summary Absent (D19):", ws1["D19"].value)
    print("Summary Rate (E19):", ws1["E19"].value)
    assert "8" in ws1["A19"].value
    assert "3" in ws1["C19"].value
    assert "5" in ws1["D19"].value
    assert "37.5%" in ws1["E19"].value

    # Signatures check
    print("Signature Block A21:", ws1["A21"].value)
    print("Signature Block A22:", repr(ws1["A22"].value))
    print("Signature Block A23:", ws1["A23"].value)
    print("Dept Head Block E21:", ws1["E21"].value)
    assert ws1["A21"].value == "أستاذ المادة المشرف:"
    assert ws1["A22"].value in (None, "")
    assert "التوقيع:" in ws1["A23"].value
    # Ensure Department Head block has been completely removed
    assert ws1["E21"].value is None
    assert ws1["E22"].value is None
    assert ws1["E23"].value is None
    print("Signatures checked successfully: ONLY Supervising Course Professor signature block retained.")

    # TEST 2: Dynamic Query Parameters Override
    print("\n--- Test 2: Dynamic Query Parameters Override ---")
    resp2 = await main.export_report(
        department="قسم تقنيات الأجهزة الطبية",
        subject="الأنظمة الطبية الذكية",
        section="الشعبة أ",
        stage="المرحلة الثالثة",
        instructor_name="د. سارة خليل",
        date="2026/11/01",
        time="09:30:00 AM",
    )
    body2 = b""
    async for chunk in resp2.body_iterator:
        body2 += chunk

    wb2 = openpyxl.load_workbook(io.BytesIO(body2))
    ws2 = wb2.active
    assert "قسم تقنيات الأجهزة الطبية" in ws2["A3"].value
    assert ws2["B5"].value == "الأنظمة الطبية الذكية"
    assert ws2["F5"].value == "الشعبة أ"
    assert ws2["A6"].value == "أستاذ المادة المشرف:"
    assert ws2["B6"].value in (None, "")
    assert ws2["D6"].value == "2026/11/01"
    assert ws2["F6"].value == "09:30:00 AM"
    assert ws2["A21"].value == "أستاذ المادة المشرف:"
    assert ws2["A22"].value in (None, "")
    assert ws2["E21"].value is None
    print("Dynamic parameter overrides & layout verification passed successfully!")

    main.SESSION.terminate()
    print("\n" + "=" * 60)
    print("ALL OPENPYXL INSTITUTIONAL REPORT TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_export_endpoint())
