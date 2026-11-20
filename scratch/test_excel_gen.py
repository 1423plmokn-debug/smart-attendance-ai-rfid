"""
Verification script for OpenPyXL Excel report generation.
Tests institutional layout, RTL sheet view, dynamic parameters,
absent student integration, summary metrics, and signatures.
"""
import io
import sys
from datetime import datetime
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

def build_excel_report(
    department: str = "هندسة تقنيات الإلكترونيك والذكاء الاصطناعي",
    subject: str = "الرؤية الحاسوبية والأنظمة الذكية",
    section: str = "الشعبة أ",
    stage: str = "المرحلة الرابعة",
    instructor_name: str = "أ.م.د. علي التميمي",
    session_dt: datetime = None,
    attendance_records: list = None,
    all_students: list = None,
) -> bytes:
    if session_dt is None:
        session_dt = datetime.now()
    if attendance_records is None:
        attendance_records = [
            {"name": "Ali Dhea", "timestamp": "2026-10-28T10:14:22", "method": "كاميرا ذكية (AI Vision)"},
            {"name": "Ali Nehad", "timestamp": "2026-10-28T10:15:05", "method": "كاميرا ذكية (AI Vision)"},
            {"name": "Baneen Talib", "timestamp": "2026-10-28T10:16:30", "method": "كاميرا ذكية (AI Vision)"},
        ]
    if all_students is None:
        all_students = [
            "Ali Dhea", "Ali Nehad", "Baneen Talib", "Enas Abdulkarim",
            "Mahdi Ali", "Mahdi Imad", "Ruaa Nadeem", "Sajjad Khazaal"
        ]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "سجل الحضور الأكاديمي"

    # Enforce Right-to-Left (RTL) mode
    ws.sheet_view.rightToLeft = True
    try:
        if hasattr(ws, "views") and hasattr(ws.views, "sheetView") and ws.views.sheetView:
            ws.views.sheetView[0].rightToLeft = True
    except Exception:
        pass
    ws.views.sheetView[0].showGridLines = True

    # Styling Palettes
    navy_dark   = "0F2942"
    navy_mid    = "1A365D"
    card_bg     = "F8FAFC"
    border_col  = "CBD5E0"
    header_fill = PatternFill("solid", fgColor=navy_dark)
    sub_fill    = PatternFill("solid", fgColor=navy_mid)
    info_lbl_bg = PatternFill("solid", fgColor="EDF2F7")
    info_val_bg = PatternFill("solid", fgColor="FFFFFF")

    thin_border = Border(
        left=Side(style="thin", color=border_col),
        right=Side(style="thin", color=border_col),
        top=Side(style="thin", color=border_col),
        bottom=Side(style="thin", color=border_col),
    )

    # 1. Main Header
    ws.merge_cells("A2:F2")
    c_main = ws["A2"]
    c_main.value = "الجامعة التقنية الوسطى / كلية البوليتكنك"
    c_main.font = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    c_main.fill = header_fill
    c_main.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 36

    # 2. Sub-Header (Dynamic Department)
    dept_display = department if department.startswith("قسم") else f"قسم {department}"
    ws.merge_cells("A3:F3")
    c_sub = ws["A3"]
    c_sub.value = f"{dept_display} — استمارة حضور وغياب الطلبة للمحاضرة الأكاديمية"
    c_sub.font = Font(name="Calibri", size=12, bold=True, color="E2E8F0")
    c_sub.fill = sub_fill
    c_sub.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[3].height = 28

    # Apply border across merged headers
    for r in (2, 3):
        for col_idx in range(1, 7):
            cell = ws.cell(row=r, column=col_idx)
            cell.border = thin_border
            if r == 2:
                cell.fill = header_fill
            else:
                cell.fill = sub_fill

    ws.row_dimensions[4].height = 10

    # 3. Session Info Block (Rows 5 & 6)
    report_date = session_dt.strftime("%Y/%m/%d")
    report_time = session_dt.strftime("%I:%M:%S %p")

    info_meta = [
        # Row 5:
        [
            ("A", "المادة الأكاديمية:", True),
            ("B", subject, False),
            ("C", "القسم / المرحلة:", True),
            ("D", f"{dept_display} ({stage})", False),
            ("E", "الشعبة:", True),
            ("F", section, False),
        ],
        # Row 6:
        [
            ("A", "أستاذ المادة:", True),
            ("B", instructor_name, False),
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

    # 4. Table Header (Row 8)
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

    # 5. Data Rows
    logged_map = {rec["name"]: rec for rec in attendance_records}
    master_roster = list(all_students)
    for rec in attendance_records:
        if rec.get("name") and rec["name"] not in master_roster:
            master_roster.append(rec["name"])

    # Separate Present vs Absent
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
            status_text = "حاضر ✓"
            status_font = Font(name="Calibri", size=10, bold=True, color="065F46")
            status_fill = PatternFill("solid", fgColor="D1FAE5")
        else:
            method = "—"
            ts = "—"
            status_text = "غائب ✗"
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
        c_sec.value = f"{dept_display} / {section}"
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

        # Apply basic formatting to columns A-E
        for col_l in ("A", "B", "C", "D", "E"):
            c = ws[f"{col_l}{cur_row}"]
            c.font = cell_font
            c.fill = row_fill
            c.border = thin_border
        c_stat.border = thin_border

    # 6. Summary Footer Block
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
        # Title Cell
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

        # Borders
        cols = [start_c] if start_c == end_c else [start_c, end_c]
        for c in cols:
            ws[f"{c}{sum_title_row}"].border = thin_border
            ws[f"{c}{sum_data_row}"].border = thin_border

    # 7. Signature Block
    sig_space_row = sum_data_row + 1
    sig_row1 = sum_data_row + 2
    sig_row2 = sum_data_row + 3
    sig_row3 = sum_data_row + 4

    ws.row_dimensions[sig_space_row].height = 18
    ws.row_dimensions[sig_row1].height = 22
    ws.row_dimensions[sig_row2].height = 22
    ws.row_dimensions[sig_row3].height = 22

    # Instructor Signature (Right Side: A-B)
    ws.merge_cells(f"A{sig_row1}:B{sig_row1}")
    ws.merge_cells(f"A{sig_row2}:B{sig_row2}")
    ws.merge_cells(f"A{sig_row3}:B{sig_row3}")

    ws[f"A{sig_row1}"].value = "أستاذ المادة المشرف:"
    ws[f"A{sig_row1}"].font = Font(name="Calibri", size=11, bold=True, color="0F2942")
    ws[f"A{sig_row1}"].alignment = Alignment(horizontal="center", vertical="center")

    ws[f"A{sig_row2}"].value = instructor_name
    ws[f"A{sig_row2}"].font = Font(name="Calibri", size=11, bold=True, color="334155")
    ws[f"A{sig_row2}"].alignment = Alignment(horizontal="center", vertical="center")

    ws[f"A{sig_row3}"].value = "التوقيع: ......................................."
    ws[f"A{sig_row3}"].font = Font(name="Calibri", size=10, color="64748B")
    ws[f"A{sig_row3}"].alignment = Alignment(horizontal="center", vertical="center")

    # Dean / Head of Dept Signature (Left Side: E-F)
    ws.merge_cells(f"E{sig_row1}:F{sig_row1}")
    ws.merge_cells(f"E{sig_row2}:F{sig_row2}")
    ws.merge_cells(f"E{sig_row3}:F{sig_row3}")

    ws[f"E{sig_row1}"].value = "رئيس القسم الأكاديمي:"
    ws[f"E{sig_row1}"].font = Font(name="Calibri", size=11, bold=True, color="0F2942")
    ws[f"E{sig_row1}"].alignment = Alignment(horizontal="center", vertical="center")

    ws[f"E{sig_row2}"].value = "رئاسة القسم / التدقيق"
    ws[f"E{sig_row2}"].font = Font(name="Calibri", size=11, bold=True, color="334155")
    ws[f"E{sig_row2}"].alignment = Alignment(horizontal="center", vertical="center")

    ws[f"E{sig_row3}"].value = "التوقيع والختم: ......................................."
    ws[f"E{sig_row3}"].font = Font(name="Calibri", size=10, color="64748B")
    ws[f"E{sig_row3}"].alignment = Alignment(horizontal="center", vertical="center")

    # Column Widths
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 38
    ws.column_dimensions["D"].width = 24
    ws.column_dimensions["E"].width = 24
    ws.column_dimensions["F"].width = 18

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

if __name__ == "__main__":
    data = build_excel_report()
    out_file = Path(__file__).parent / "test_out.xlsx"
    out_file.write_bytes(data)
    print(f"Generated test_out.xlsx ({len(data)} bytes) successfully!")
