# coding: utf-8
"""Generate the first thesis draft for the face attendance project.

The document intentionally separates verified engineering evidence from pending
experiments. It uses only anonymised metrics and contains no student records,
face images, embeddings, passwords, or local account names.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "毕业设计论文初稿_实验室可信通行版.docx"
ASSET_DIR = ROOT / "docs" / "thesis_assets"
METRICS_PATH = ROOT / "evaluation_report" / "sface_metrics.json"

TITLE = "基于深度学习的高校实验室人员全生命周期可信通行平台的设计与实现"
SUBTITLE = "面向多人实时识别、动态权限与可验证销毁的一体化身份治理"

# narrative_proposal preset + named academic Chinese overrides:
# A4 page, SimSun/Times New Roman body, 12 pt, 1.5 lines.
INK = "24384E"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
MUTED = "65758B"
LIGHT_BLUE = "EAF2FA"
LIGHT_GRAY = "F4F6F9"
LINE = "C8D4E3"
RED = "9B1C1C"
GOLD = "8A6500"
WHITE = "FFFFFF"
CONTENT_DXA = 9070
TABLE_INDENT_DXA = 100


def set_font(run, east="宋体", latin="Times New Roman", size=12,
             bold=None, color=None, italic=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run._element.rPr.rFonts.set(qn("w:ascii"), latin)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), latin)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=120, bottom=90, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start),
                        ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_geometry(table, widths_dxa, indent_dxa=TABLE_INDENT_DXA):
    total = sum(widths_dxa)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[min(index, len(widths_dxa) - 1)]
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_borders(table, color=LINE, size=6):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), str(size))
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def add_page_field(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instr, separate, text, end):
        run._r.append(node)
    set_font(run, east="宋体", latin="Times New Roman", size=9, color=MUTED)


def create_numbering(doc):
    numbering = doc.part.numbering_part.element

    def add_num(abstract_id, num_id, fmt, text, left, hanging, font):
        abstract = OxmlElement("w:abstractNum")
        abstract.set(qn("w:abstractNumId"), str(abstract_id))
        multi = OxmlElement("w:multiLevelType")
        multi.set(qn("w:val"), "singleLevel")
        abstract.append(multi)
        level = OxmlElement("w:lvl")
        level.set(qn("w:ilvl"), "0")
        start = OxmlElement("w:start")
        start.set(qn("w:val"), "1")
        num_fmt = OxmlElement("w:numFmt")
        num_fmt.set(qn("w:val"), fmt)
        lvl_text = OxmlElement("w:lvlText")
        lvl_text.set(qn("w:val"), text)
        suff = OxmlElement("w:suff")
        suff.set(qn("w:val"), "tab")
        p_pr = OxmlElement("w:pPr")
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "num")
        tab.set(qn("w:pos"), str(left))
        tabs.append(tab)
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), str(left))
        ind.set(qn("w:hanging"), str(hanging))
        spacing = OxmlElement("w:spacing")
        spacing.set(qn("w:after"), "80")
        spacing.set(qn("w:line"), "290")
        spacing.set(qn("w:lineRule"), "auto")
        p_pr.extend((tabs, ind, spacing))
        r_pr = OxmlElement("w:rPr")
        r_fonts = OxmlElement("w:rFonts")
        r_fonts.set(qn("w:ascii"), font)
        r_fonts.set(qn("w:hAnsi"), font)
        r_fonts.set(qn("w:eastAsia"), font)
        r_pr.append(r_fonts)
        level.extend((start, num_fmt, lvl_text, suff, p_pr, r_pr))
        abstract.append(level)
        numbering.append(abstract)
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(num_id))
        abstract_ref = OxmlElement("w:abstractNumId")
        abstract_ref.set(qn("w:val"), str(abstract_id))
        num.append(abstract_ref)
        numbering.append(num)

    add_num(90, 90, "bullet", "●", 540, 260, "宋体")
    add_num(91, 91, "decimal", "%1.", 540, 260, "Times New Roman")
    add_num(92, 92, "decimal", "[%1]", 620, 360, "Times New Roman")


def set_num(paragraph, num_id):
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        p_pr.append(num_pr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    num_pr.extend((ilvl, num))


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Pt(24)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.5

    heading_tokens = {
        "Heading 1": (16, BLUE, 18, 10, "黑体"),
        "Heading 2": (14, BLUE, 12, 6, "黑体"),
        "Heading 3": (12, DARK_BLUE, 8, 4, "黑体"),
    }
    for name, (size, color, before, after, east) in heading_tokens.items():
        style = styles[name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), east)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.first_line_indent = Pt(0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.2
        style.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = "Times New Roman"
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    caption.font.size = Pt(10.5)
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.keep_with_next = False


def configure_page(doc):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.header_distance = Cm(1.5)
    section.footer_distance = Cm(1.5)
    section.different_first_page_header_footer = True

    header_p = section.header.paragraphs[0]
    header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_p.paragraph_format.space_after = Pt(0)
    run = header_p.add_run(TITLE + "（初稿）")
    set_font(run, east="宋体", latin="Times New Roman", size=8.5, color=MUTED)
    footer_p = section.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_p.paragraph_format.space_before = Pt(0)
    add_page_field(footer_p)


def add_body(doc, text, bold_prefix=None, italic=False, no_indent=False,
             keep_with_next=False):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = keep_with_next
    if no_indent:
        p.paragraph_format.first_line_indent = Pt(0)
    if bold_prefix and text.startswith(bold_prefix):
        lead = p.add_run(bold_prefix)
        set_font(lead, east="黑体", size=12, bold=True, color=INK)
        rest = p.add_run(text[len(bold_prefix):])
        set_font(rest, size=12, italic=italic, color=INK)
    else:
        run = p.add_run(text)
        set_font(run, size=12, italic=italic, color=INK)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(20)
    p.paragraph_format.hanging_indent = Pt(20)
    run = p.add_run("•  " + text)
    set_font(run, size=11.5, color=INK)
    return p


def add_numbered(doc, text, number):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(28)
    p.paragraph_format.hanging_indent = Pt(28)
    run = p.add_run(f"（{number}）" + text)
    set_font(run, size=11.5, color=INK)
    return p


def add_callout(doc, label, text, color=BLUE, fill=LIGHT_BLUE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [CONTENT_DXA])
    set_table_borders(table, color=fill, size=4)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    label_run = p.add_run(label + "：")
    set_font(label_run, east="黑体", size=11, bold=True, color=color)
    text_run = p.add_run(text)
    set_font(text_run, size=11, color=INK)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(0)
    after.paragraph_format.line_spacing = 0.4
    return table


def add_equation(doc, equation, number):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run(equation)
    set_font(run, east="Cambria Math", latin="Cambria Math", size=11.5, color=INK)
    number_run = p.add_run("    " + number)
    set_font(number_run, east="宋体", latin="Times New Roman", size=11, color=MUTED)


def add_table(doc, headers, rows, widths, caption=None, note=None,
              alignments=None):
    if caption:
        p = doc.add_paragraph(style="Caption")
        p.paragraph_format.keep_with_next = True
        run = p.add_run(caption)
        set_font(run, east="宋体", size=10.5, color=MUTED)
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    set_table_borders(table)
    set_repeat_table_header(table.rows[0])
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        set_cell_shading(cell, LIGHT_GRAY)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(str(header))
        set_font(run, east="黑体", size=10, bold=True, color=INK)
    for row_data in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_data):
            p = cells[index].paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            alignment = (alignments[index] if alignments else
                         WD_ALIGN_PARAGRAPH.CENTER)
            p.alignment = alignment
            run = p.add_run(str(value))
            set_font(run, size=9.5, color=INK)
        set_table_geometry(table, widths)
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(note)
        set_font(run, size=9.5, color=MUTED, italic=True)
    return table


def add_picture(doc, path, caption, width_cm=15.8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = doc.add_paragraph(style="Caption")
    run = cap.add_run(caption)
    set_font(run, east="宋体", size=10.5, color=MUTED)


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.page_break_before = True
    return p


FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")


def image_font(size, bold=False):
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered_multiline(draw, xy, text, font, fill, spacing=8):
    x, y = xy
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing,
                                  align="center")
    width, height = box[2] - box[0], box[3] - box[1]
    draw.multiline_text((x - width / 2, y - height / 2), text, font=font,
                        fill=fill, spacing=spacing, align="center")


def draw_arrow(draw, start, end, fill="#4B78A5", width=4):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 16
    left = (x2 - length * math.cos(angle - math.pi / 6),
            y2 - length * math.sin(angle - math.pi / 6))
    right = (x2 - length * math.cos(angle + math.pi / 6),
             y2 - length * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), left, right], fill=fill)


def save_diagram(path, boxes, arrows, title, figsize=(11, 5.6)):
    width = 1800
    height = int(width * figsize[1] / figsize[0])
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = image_font(42, bold=True)
    box_font = image_font(29)
    label_font = image_font(23)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_box[2] - title_box[0]
    draw.text(((width - title_width) / 2, 24), title, font=title_font,
              fill="#24384E")
    top_pad = 76
    usable_height = height - top_pad - 24
    palette = ["#EAF2FA", "#E8F5F3", "#FFF5E6", "#F2EEFB", "#F4F6F9"]
    for index, (x, y, w, h, text) in enumerate(boxes):
        left = int(x * width)
        top = int(top_pad + (1 - y - h) * usable_height)
        right = int((x + w) * width)
        bottom = int(top_pad + (1 - y) * usable_height)
        draw.rounded_rectangle((left, top, right, bottom), radius=22,
                               fill=palette[index % len(palette)],
                               outline="#6E8FB1", width=3)
        draw_centered_multiline(draw, ((left + right) / 2, (top + bottom) / 2),
                                text, box_font, "#24384E", spacing=8)
    for x1, y1, x2, y2, label in arrows:
        start = (int(x1 * width), int(top_pad + (1 - y1) * usable_height))
        end = (int(x2 * width), int(top_pad + (1 - y2) * usable_height))
        draw_arrow(draw, start, end)
        if label:
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 25)
            draw_centered_multiline(draw, mid, label, label_font, "#65758B", spacing=3)
    image.save(path, quality=95)


def build_assets(metrics):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    architecture = ASSET_DIR / "architecture.png"
    boxes = [
        (0.03, 0.58, 0.16, 0.18, "可信通行终端\n摄像头 / 门禁点"),
        (0.25, 0.58, 0.18, 0.18, "Flask 接口层\n校验 / 会话 / CSRF"),
        (0.49, 0.58, 0.20, 0.18, "识别决策引擎\nYuNet + SFace\nMiniFASNetV2"),
        (0.75, 0.58, 0.20, 0.18, "可信访问策略\n状态 / 成员 / 授权"),
        (0.25, 0.18, 0.18, 0.18, "治理管理端\n人员 / 实验室 / 门禁"),
        (0.49, 0.18, 0.20, 0.18, "生命周期数据层\n模板 / 权限 / 审计"),
        (0.75, 0.18, 0.20, 0.18, "可视化与销毁\nECharts / 回执"),
    ]
    arrows = [
        (0.19, 0.67, 0.25, 0.67, "二进制帧"),
        (0.43, 0.67, 0.49, 0.67, "调用"),
        (0.69, 0.67, 0.75, 0.67, "最终决策"),
        (0.34, 0.58, 0.34, 0.36, "管理请求"),
        (0.59, 0.58, 0.59, 0.36, "审计落库"),
        (0.85, 0.58, 0.85, 0.36, "统计"),
        (0.43, 0.27, 0.49, 0.27, "CRUD"),
        (0.69, 0.27, 0.75, 0.27, "聚合"),
    ]
    save_diagram(architecture, boxes, arrows, "系统总体架构")

    pipeline = ASSET_DIR / "recognition_pipeline.png"
    boxes = [
        (0.02, 0.62, 0.14, 0.16, "摄像头帧\n输入校验"),
        (0.20, 0.62, 0.14, 0.16, "YuNet\n多人检测"),
        (0.38, 0.62, 0.14, 0.16, "质量评分\n关键帧调度"),
        (0.56, 0.62, 0.14, 0.16, "轨迹关联\n时序特征融合"),
        (0.74, 0.62, 0.14, 0.16, "SFace\n开放集判定"),
        (0.74, 0.24, 0.14, 0.16, "静默活体\n风险分级"),
        (0.56, 0.24, 0.14, 0.16, "必要时随机\n眨眼或转头"),
        (0.38, 0.24, 0.14, 0.16, "多帧投票\n身份冲突检查"),
        (0.20, 0.24, 0.14, 0.16, "权限决策\n状态 / 区域 / 时段"),
        (0.02, 0.24, 0.14, 0.16, "通行与考勤\n审计落库"),
    ]
    arrows = [
        (0.16, 0.70, 0.20, 0.70, ""), (0.34, 0.70, 0.38, 0.70, ""),
        (0.52, 0.70, 0.56, 0.70, ""), (0.70, 0.70, 0.74, 0.70, ""),
        (0.81, 0.62, 0.81, 0.40, ""), (0.74, 0.32, 0.70, 0.32, "中风险"),
        (0.56, 0.32, 0.52, 0.32, ""), (0.38, 0.32, 0.34, 0.32, ""),
        (0.20, 0.32, 0.16, 0.32, ""),
    ]
    save_diagram(pipeline, boxes, arrows, "单帧输入到实验室可信通行结果的决策链", figsize=(12, 5.2))

    data_model = ASSET_DIR / "data_model.png"
    boxes = [
        (0.03, 0.70, 0.20, 0.14, "User\n人员与生命周期"),
        (0.40, 0.70, 0.20, 0.14, "OrganizationUnit\n实验室组织"),
        (0.77, 0.70, 0.20, 0.14, "AccessPoint\n门禁点"),
        (0.03, 0.39, 0.20, 0.14, "FaceTemplate\n独立生物模板"),
        (0.40, 0.39, 0.20, 0.14, "LabMembership\n成员关系"),
        (0.77, 0.39, 0.20, 0.14, "AccessGrant\n时段授权"),
        (0.03, 0.08, 0.20, 0.14, "RecognitionEvent\n识别证据"),
        (0.40, 0.08, 0.20, 0.14, "AccessEvent\n通行决策"),
        (0.77, 0.08, 0.20, 0.14, "ErasureTask\n匿名销毁回执"),
    ]
    arrows = [
        (0.13, 0.70, 0.13, 0.53, "1:1"),
        (0.23, 0.77, 0.40, 0.77, "归属"),
        (0.50, 0.70, 0.50, 0.53, "1:N"),
        (0.60, 0.77, 0.77, 0.77, "1:N"),
        (0.87, 0.70, 0.87, 0.53, "1:N"),
        (0.23, 0.46, 0.40, 0.46, "人员"),
        (0.60, 0.46, 0.77, 0.46, "授权"),
        (0.13, 0.39, 0.13, 0.22, "证据"),
        (0.23, 0.15, 0.40, 0.15, "决策"),
        (0.60, 0.15, 0.77, 0.15, "离组"),
    ]
    save_diagram(data_model, boxes, arrows, "人员全生命周期可信通行核心数据模型", figsize=(10.5, 5.8))

    performance = ASSET_DIR / "performance.png"
    image = Image.new("RGB", (1760, 930), "white")
    draw = ImageDraw.Draw(image)
    title_font = image_font(40, bold=True)
    axis_font = image_font(25)
    value_font = image_font(23)
    legend_font = image_font(24)
    labels = ["中位耗时", "P95", "平均耗时"]
    dlib = [286.61, 390.25, 299.18]
    fast = [16.65, 32.85, 19.97]
    title = "45 个样本下两条人脸流水线耗时对比"
    tbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((1760 - (tbox[2] - tbox[0])) / 2, 25), title,
              font=title_font, fill="#24384E")
    left, top, right, bottom = 145, 135, 1650, 760
    max_value = 420.0
    for tick in range(0, 401, 100):
        y = bottom - int((tick / max_value) * (bottom - top))
        draw.line((left, y, right, y), fill="#E0E7EF", width=2)
        draw.text((62, y - 14), str(tick), font=axis_font, fill="#65758B")
    draw.line((left, top, left, bottom), fill="#8093A8", width=3)
    draw.line((left, bottom, right, bottom), fill="#8093A8", width=3)
    group_width = (right - left) / len(labels)
    bar_width = 100
    for index, label in enumerate(labels):
        center = left + group_width * (index + 0.5)
        for value, offset, color in ((dlib[index], -58, "#AAB7C6"),
                                     (fast[index], 58, "#3478F6")):
            x1 = int(center + offset - bar_width / 2)
            x2 = x1 + bar_width
            y1 = bottom - int((value / max_value) * (bottom - top))
            draw.rounded_rectangle((x1, y1, x2, bottom), radius=9, fill=color)
            value_text = f"{value:.2f}"
            vbox = draw.textbbox((0, 0), value_text, font=value_font)
            draw.text((x1 + (bar_width - (vbox[2] - vbox[0])) / 2, y1 - 32),
                      value_text, font=value_font, fill="#45576B")
        lbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text((center - (lbox[2] - lbox[0]) / 2, bottom + 20), label,
                  font=axis_font, fill="#24384E")
    draw.text((20, 420), "耗时 / ms", font=axis_font, fill="#65758B")
    draw.rounded_rectangle((520, 840, 555, 870), radius=5, fill="#AAB7C6")
    draw.text((568, 838), "dlib HOG + ResNet", font=legend_font, fill="#45576B")
    draw.rounded_rectangle((1010, 840, 1045, 870), radius=5, fill="#3478F6")
    draw.text((1058, 838), "YuNet + SFace", font=legend_font, fill="#45576B")
    image.save(performance, quality=95)

    score_plot = ASSET_DIR / "score_distribution.png"
    genuine = metrics["scores"]["genuine"]
    impostor = metrics["scores"]["impostor"]
    threshold = metrics["threshold"]
    image = Image.new("RGB", (1760, 930), "white")
    draw = ImageDraw.Draw(image)
    title = "小样本 SFace 相似度分布（仅作功能性基线）"
    title_font = image_font(40, bold=True)
    axis_font = image_font(25)
    legend_font = image_font(23)
    tbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((1760 - (tbox[2] - tbox[0])) / 2, 25), title,
              font=title_font, fill="#24384E")
    left, top, right, bottom = 180, 145, 1650, 760
    min_x, max_x = -0.2, 0.9
    def map_x(value):
        return int(left + (value - min_x) / (max_x - min_x) * (right - left))
    for tick_int in range(-2, 10):
        tick = tick_int / 10
        x = map_x(tick)
        draw.line((x, top, x, bottom), fill="#E1E8EF", width=2)
        label = f"{tick:.1f}"
        box = draw.textbbox((0, 0), label, font=axis_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 20), label,
                  font=axis_font, fill="#65758B")
    draw.line((left, bottom, right, bottom), fill="#8093A8", width=3)
    draw.text((55, 292), "正样本", font=axis_font, fill="#24384E")
    draw.text((55, 575), "负样本", font=axis_font, fill="#24384E")
    for index, value in enumerate(genuine):
        x = map_x(value)
        y = 310 + (index % 2) * 25
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill="#13A986")
    for index, value in enumerate(impostor):
        x = map_x(value)
        y = 590 + ((index % 5) - 2) * 22
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#DF5A69")
    tx = map_x(threshold)
    for y in range(top, bottom, 20):
        draw.line((tx, y, tx, min(y + 10, bottom)), fill="#8A6500", width=4)
    draw.text((tx + 12, top + 6), f"阈值 {threshold:.2f}", font=legend_font,
              fill="#8A6500")
    xlabel = "余弦相似度"
    xbox = draw.textbbox((0, 0), xlabel, font=axis_font)
    draw.text(((left + right - (xbox[2] - xbox[0])) / 2, 835), xlabel,
              font=axis_font, fill="#24384E")
    image.save(score_plot, quality=95)

    return {
        "architecture": architecture,
        "pipeline": pipeline,
        "data_model": data_model,
        "performance": performance,
        "score_plot": score_plot,
    }


def add_cover(doc):
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run("本科毕业设计（论文）")
    set_font(run, east="黑体", latin="Arial", size=18, bold=True, color=MUTED)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(40)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(TITLE)
    set_font(run, east="黑体", latin="Arial", size=25, bold=True, color=INK)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(46)
    run = p.add_run(SUBTITLE)
    set_font(run, east="宋体", latin="Times New Roman", size=13, color=BLUE)

    fields = [
        ("学校", "[待填写：学校名称]"),
        ("学院", "[待填写：学院名称]"),
        ("专业", "[待填写：专业名称]"),
        ("学生姓名", "[待填写：姓名]"),
        ("学号", "[待填写：学号]"),
        ("指导教师", "[待填写：导师姓名及职称]"),
        ("完成日期", "2026 年 [待填写] 月"),
    ]
    table = doc.add_table(rows=len(fields), cols=2)
    set_table_geometry(table, [2300, 5000], indent_dxa=900)
    # Cover metadata has no visible borders: a named cover-layout exception.
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), "nil")
        borders.append(tag)
    tbl_pr.append(borders)
    for row, (label, value) in zip(table.rows, fields):
        p1 = row.cells[0].paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.first_line_indent = Pt(0)
        r1 = p1.add_run(label + "：")
        set_font(r1, east="黑体", size=12, bold=True, color=INK)
        p2 = row.cells[1].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p2.paragraph_format.first_line_indent = Pt(0)
        r2 = p2.add_run(value)
        set_font(r2, size=12, color=INK)
    doc.add_page_break()


def add_abstracts(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("摘  要")
    set_font(r, east="黑体", size=18, bold=True, color=INK)

    abstract = (
        "针对传统人脸签到系统只关注识别结果、缺少人员生命周期治理、区域权限校验和离组隐私清理的问题，"
        "本文设计并实现了一套基于深度学习的高校实验室人员全生命周期可信通行平台。平台以 Flask 为服务端框架，"
        "采用 YuNet 完成人脸检测，使用 SFace 提取 128 维归一化人脸特征，并以 MiniFASNetV2 实现 RGB 静默活体检测。"
        "为适应普通 CPU 与浏览器摄像头场景，本文构建了质量驱动关键帧调度、基于空间位置的多人轨迹关联、质量加权时序特征融合、"
        "相似度与候选间隔双阈值开放集判定，以及静默优先、风险不确定时触发单次随机动作的自适应活体策略。"
        "在识别之后，平台继续校验人员在组状态、实验室成员关系、门禁点授权及有效时段，实现“识别成功不等于允许通行”的最小权限决策。"
        "人员离组时，系统原子化撤销权限并物理删除档案、人脸模板和关联关系，只保留不可反查身份的审计事件与销毁回执，从而形成从入组、授权、通行到离组销毁的闭环。"
    )
    add_body(doc, abstract)
    abstract2 = (
        "在 Intel Core i7-12650H、Windows 11 和 Python 3.11 环境下，对 15 张图像分别重复 3 次，共获得 45 个流水线样本。"
        "结果表明，YuNet 与 SFace 流水线的中位耗时为 16.65 ms，P95 为 32.85 ms；传统 dlib HOG 与 ResNet 流水线的中位耗时为 "
        "286.61 ms，前者中位耗时加速 17.22 倍。预热后的 Web 完整请求约为 42～47 ms，MiniFASNetV2 单脸推理约为 1.4～1.6 ms。"
        "系统已通过 59 项自动化测试，其中包含门禁最小权限、暂停阻断和离组物理销毁测试。由于当前真实样本仅覆盖 1 个注册身份、2 个正样本和 11 个负样本，本文将相似度实验明确限定为功能性基线，"
        "不据此外推多人识别准确率；多人并发、照片/视频重放攻击和时序融合消融实验将在后续采集授权数据后补充。"
    )
    add_body(doc, abstract2)
    add_body(doc, "关键词：人脸识别；高校实验室；人员全生命周期；可信通行；活体检测；动态权限；可验证销毁",
             bold_prefix="关键词：", no_indent=True)

    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("ABSTRACT")
    set_font(r, east="Arial", latin="Arial", size=18, bold=True, color=INK)
    en1 = (
        "Conventional face attendance systems often stop at identity recognition and omit personnel lifecycle governance, zone-specific authorisation, and privacy-preserving offboarding. This thesis therefore designs and implements a deep-learning-based trusted access platform for the complete personnel lifecycle of university laboratories. The server is built with Flask. YuNet is used for multi-face detection, SFace produces L2-normalized 128-dimensional embeddings, and MiniFASNetV2 provides passive RGB presentation attack detection. The engineering pipeline combines explainable face-quality scheduling, spatial multi-face track association, quality-weighted temporal feature fusion, open-set recognition with similarity and runner-up-margin thresholds, and a risk-adaptive liveness policy. After recognition, the platform evaluates active status, laboratory membership, gate grants, and validity windows. Offboarding atomically revokes permissions and physically removes identity records and biometric templates while retaining anonymous audit and erasure receipts."
    )
    en2 = (
        "On an Intel Core i7-12650H under Windows 11 and Python 3.11, 15 images were processed three times, yielding 45 pipeline samples. YuNet plus SFace achieved a median latency of 16.65 ms and a P95 latency of 32.85 ms, compared with 286.61 ms median latency for the dlib HOG plus ResNet baseline, corresponding to a 17.22-fold median speed-up. Warm Web requests took approximately 42–47 ms, and MiniFASNetV2 inference took approximately 1.4–1.6 ms per face. The system passed 59 automated tests, including least-privilege access, suspension denial, and physical offboarding erasure tests. Because the current identity evaluation contains only one enrolled identity, two genuine probes, and eleven impostor probes, the reported threshold results are treated strictly as a functional baseline. Multi-person concurrency, photo/video replay attacks, and temporal-fusion ablation experiments are reserved for a later authorised data collection stage."
    )
    for text in (en1, en2):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(24)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        set_font(r, east="Times New Roman", latin="Times New Roman", size=11.5, color=INK)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run("KEY WORDS: ")
    set_font(r, east="Arial", latin="Arial", size=11.5, bold=True, color=INK)
    r = p.add_run("face recognition; university laboratory; personnel lifecycle; trusted access; liveness detection; dynamic authorisation; verifiable erasure")
    set_font(r, east="Times New Roman", latin="Times New Roman", size=11.5, color=INK)
    doc.add_page_break()


def add_toc_field(paragraph):
    """Insert a Word TOC field that can be updated by Word/LibreOffice."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    fallback = OxmlElement("w:t")
    fallback.text = "目录将在打开文档时自动更新"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instr, separate, fallback, end))


def add_toc_placeholder(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run("目  录")
    set_font(r, east="黑体", size=18, bold=True, color=INK)
    placeholder = doc.add_paragraph()
    placeholder.paragraph_format.first_line_indent = Pt(0)
    add_toc_field(placeholder)
    doc.add_page_break()


def add_chapter_one(doc):
    heading(doc, "第1章 绪论", 1)
    heading(doc, "1.1 研究背景与意义", 2)
    add_body(doc, "高校实验室同时具有人员流动频繁、空间与设备权限细分、生物特征敏感以及安全责任可追溯等特点。学生、教师、访客和合作人员从入组到离组，需要经历建档、实验室归属、分区授权、日常通行、状态暂停和数据清理。传统纸质登记、校园卡和独立人脸门禁通常分散管理：卡片可能转借，人脸设备只回答“是谁”，却无法解释“此人当前为何可以进入”，离组后也容易遗留档案、权限和人脸模板。")
    add_body(doc, "早期人脸识别系统主要依赖手工特征或子空间方法。随着卷积神经网络和大规模数据集的发展，人脸检测、对齐和特征表示的鲁棒性不断提高[1-7]。然而，从“模型可以识别人脸”到“平台可以可信决定实验室通行”之间仍存在显著工程鸿沟：浏览器帧具有抖动、模糊和曝光变化，多人同时出现时必须维持独立轨迹，未知人员不能被强行分类，照片或屏幕重放需要被阻断，识别成功后还必须校验人员状态、组织归属、区域授权与有效时段。")
    add_body(doc, "因此，本文不以重新训练大型基础网络为目标，而是围绕实时性、安全性、多人并发、最小权限和隐私可证明性，对预训练深度学习模型进行组合、调度并嵌入人员全生命周期。该定位既符合普通笔记本 CPU 的答辩演示条件，也使系统创新从单点算法优化提升为“身份识别—权限决策—离组销毁”的完整治理闭环。")

    heading(doc, "1.2 国内外研究现状", 2)
    heading(doc, "1.2.1 人脸检测与特征表示", 3)
    add_body(doc, "Viola-Jones 级联检测器通过积分图、AdaBoost 特征选择和级联分类器实现了早期实时目标检测[2]；MTCNN 将人脸检测与关键点定位作为级联多任务学习问题处理[3]。近年来，轻量化检测器进一步面向边缘设备优化。YuNet 采用无锚框设计和轻量主干网络，强调模型规模、检测精度与毫秒级推理之间的平衡[7]，适合本系统的 CPU 部署场景。")
    add_body(doc, "在人脸特征表示方面，FaceNet 将人脸映射到紧凑嵌入空间，使距离可直接反映相似性[4]；ArcFace 通过加性角度间隔提升类间可分性[5]；SFace 在超球面上以 Sigmoid 函数重新缩放类内和类间梯度，以缓解低质量样本和标签噪声导致的过拟合[6]。本文使用 OpenCV 提供的 SFace 预训练模型完成特征提取，不把预训练模型在公开基准上的结果等同于本系统现场性能。")

    heading(doc, "1.2.2 开放集识别与多目标跟踪", 3)
    add_body(doc, "实验室通行属于典型开放集识别问题：摄像头可能拍到未注册人员、访客或路人。开放集研究指出，测试阶段可能出现训练或注册阶段未知的类别，系统需要具备拒识能力[8]。仅选择相似度最高的注册身份会把陌生人强制归类，因此本文同时约束最高相似度和第一、第二候选之间的间隔。")
    add_body(doc, "多人画面还要求在连续帧之间保持身份状态。SORT 等在线多目标跟踪方法表明，检测质量与高效数据关联是实时跟踪的重要基础[9]。考虑到通行画面持续时间短、摄像头固定且目标数量有限，本文没有引入完整卡尔曼滤波器，而是采用 IoU 与归一化中心距离进行轻量一一关联，并将身份投票、活体状态和融合特征绑定到轨迹。")

    heading(doc, "1.2.3 人脸呈现攻击检测", 3)
    add_body(doc, "人脸系统可能遭受打印照片、手机静态照片、屏幕视频重放甚至面具等呈现攻击。OULU-NPU 等数据集通过不同设备、光照、背景和攻击介质评估人脸呈现攻击检测的泛化能力[10]。静默活体无需用户配合，但 RGB 纹理方法容易受到摄像头与场景域差异影响；动作活体具有较强可解释性，却会增加交互等待。MiniFASNetV2 是 Silent-Face-Anti-Spoofing 项目中的轻量模型，公开资料给出约 0.435M 参数和 0.081G FLOPs[11]。本文采用“静默优先、风险不确定时单次随机动作”的组合策略，在体验与安全之间取得平衡。")

    heading(doc, "1.3 研究目标与主要内容", 2)
    add_body(doc, "本文目标是在普通笔记本电脑上实现一套可离线答辩演示、支持同画面多人独立识别、具备呈现攻击防护，并覆盖高校实验室人员入组、授权、通行、暂停和离组销毁的应用平台。围绕该目标，主要完成以下工作：")
    for item in [
        "构建 YuNet、SFace 与 MiniFASNetV2 的 CPU 推理链路，并保留 dlib 兼容回退路径；",
        "设计由清晰度、曝光、脸部尺寸、姿态和检测置信度构成的质量评分，用于关键帧调度；",
        "设计多人空间轨迹、质量加权时序特征融合、多帧投票和开放集双阈值识别；",
        "设计风险自适应活体状态机，并对同一身份同帧多次出现的冲突进行阻断；",
        "建立人员、实验室、成员关系、门禁点、时段授权、识别事件与通行事件的数据闭环；",
        "设计离组原子化撤权与生物模板物理删除流程，生成匿名销毁回执并保留必要审计；",
        "保留课程考勤联动，实现浅色治理驾驶舱、隐私中心、实验中心和 CSV 导出，并完成自动化测试。",
    ]:
        add_bullet(doc, item)

    heading(doc, "1.4 论文工作的创新点", 2)
    add_callout(doc, "创新定位", "本文不宣称提出新的基础人脸网络，而是提出“识别成功不等于允许通行”的实验室全生命周期可信身份治理框架。", color=BLUE)
    add_numbered(doc, "质量驱动的时序识别：将透明可解释的质量评分同时用于特征提取门控、低质量阈值收紧和轨迹特征融合，降低模糊帧对最终身份的干扰。", 1)
    add_numbered(doc, "风险自适应活体：融合静默活体概率、身份置信度、图像质量和证据稳定度，仅在证据不充分时触发一次随机眨眼或转头，避免每次通行核验都长时间等待。", 2)
    add_numbered(doc, "识别与权限解耦：身份及活体通过后，继续校验在组状态、实验室成员关系、门禁授权和有效时段，默认拒绝缺少显式授权的区域访问。", 3)
    add_numbered(doc, "多人身份冲突防护：当同一注册身份在同一帧的两条轨迹上同时出现时，两条轨迹均不允许通行，并记录身份冲突风险，降低真人与其照片同时入镜造成的误放行风险。", 4)
    add_numbered(doc, "可验证离组销毁：以单次事务撤销权限并删除个人档案、人脸模板和成员关系，同时对历史事件去标识化并生成不可反查身份的销毁回执。", 5)

    heading(doc, "1.5 论文结构", 2)
    add_body(doc, "全文共七章。第1章说明研究背景、现状、目标和创新点；第2章介绍人脸检测、特征表示、开放集识别、轨迹关联和活体检测等理论基础；第3章进行需求分析和系统总体设计；第4章详细阐述核心算法与决策策略；第5章介绍系统实现；第6章给出现有实验结果、测试结果和待补实验方案；第7章总结全文并讨论后续工作。")


def add_chapter_two(doc):
    heading(doc, "第2章 相关理论与关键技术", 1)
    heading(doc, "2.1 深度学习人脸识别基本流程", 2)
    add_body(doc, "完整的人脸识别流程通常包括图像获取、人脸检测、关键点定位与对齐、特征提取、特征归一化、相似度计算和身份判定。检测阶段解决“画面中有哪些人脸”，识别阶段解决“某张人脸是否属于注册库中的某个身份”。系统工程中还需增加质量控制、轨迹管理、活体检测和最终业务约束。")

    heading(doc, "2.2 YuNet 人脸检测", 2)
    add_body(doc, "YuNet 是面向边缘设备设计的轻量无锚框人脸检测器，其输出包含边界框、双眼、鼻尖和嘴角共 5 个关键点以及检测置信度[7]。本文在最长边超过 640 像素时等比例缩小检测图像，检测完成后将边界框与关键点映射回原分辨率。该策略能够减少 DNN 推理开销，同时保留原图用于对齐、特征提取和活体裁剪。")
    add_body(doc, "OpenCV 的 FaceDetectorYN 接口允许动态设置输入尺寸，并提供分数阈值、非极大值抑制阈值和 Top-K 参数；本文使用 0.75 的检测阈值与 0.30 的 NMS 阈值。模型对象具备内部状态，因此实现中以可重入锁保护输入尺寸设置和推理过程，避免多线程请求相互覆盖。")

    heading(doc, "2.3 SFace 特征与余弦相似度", 2)
    add_body(doc, "SFace 模型将对齐后的人脸映射为 128 维特征向量。本文先进行 L2 归一化，再通过点积计算余弦相似度。设注册特征为 e_i，待识别特征为 e_p，二者均满足二范数为 1，则相似度为：")
    add_equation(doc, "s_i = e_i^T e_p", "（2-1）")
    add_body(doc, "相似度越大表示身份越接近。与直接训练分类器相比，嵌入式识别可以在不重新训练网络的情况下增加注册身份，适合人员变化频繁的实验室管理场景。本文同时计算第一候选得分 s_(1) 与第二候选得分 s_(2) 的差值 m：")
    add_equation(doc, "m = s_(1) - s_(2)", "（2-2）")
    add_body(doc, "当最高相似度未达到阈值，或候选间隔不足时，系统拒绝给出身份。单人注册库不存在第二候选，其候选间隔在数学上视为无穷，但接口层以空值返回，避免 JSON 中出现非有限数值。")

    heading(doc, "2.4 多目标轨迹关联", 2)
    add_body(doc, "本文的轨迹关联遵循 tracking-by-detection 思路[9]。每一帧首先独立检测所有人脸，再将当前边界框与会话中未超过生存时间的历史轨迹计算 IoU 和归一化中心距离。若 IoU 不小于 0.10，或中心距离不大于 0.70，则构成候选匹配；候选得分定义为 2IoU + max(0, 1-d)，按得分从高到低贪心完成一一匹配。未匹配人脸创建新轨迹，超过 2 s 未出现的轨迹被清理。")
    add_body(doc, "该关联算法没有对快速运动和长时间遮挡建模，理论能力弱于 SORT 或 DeepSORT，但它不需要额外跟踪网络，状态规模小，适用于固定摄像头、短距离排队通行和人数有限的答辩场景。后续真实多人测试将重点统计漏检、轨迹切换和身份抖动。")

    heading(doc, "2.5 活体检测技术", 2)
    add_body(doc, "活体检测可分为静默检测与交互式动作检测。静默检测从纹理、频域或深度等线索判断输入是否为真人，交互式检测则要求用户完成眨眼、转头或张嘴等随机动作。RGB 静默检测部署简单，但对设备与光照变化敏感；动作检测对照片攻击更直观，却会增加交互时间。")
    add_body(doc, "本文使用 MiniFASNetV2 的 ONNX 模型，通过 OpenCV DNN 在 CPU 上推理，输出真人类别概率。动作检测使用 dlib 68 点关键点：眨眼以双眼 EAR 由睁眼阈值下降至闭眼阈值后再次回升为依据；转头以鼻尖相对双眼中心的归一化水平位移变化为依据。EAR 思路来源于基于关键点的实时眨眼检测研究[12]。")

    heading(doc, "2.6 Web 与数据持久化技术", 2)
    add_body(doc, "系统使用 Flask 构建路由、会话与权限，Flask-WTF 提供 CSRF 防护；SQLAlchemy 与 Alembic 管理持久化及迁移。通行终端通过 getUserMedia 采集画面并以二进制 JPEG 上传；Bootstrap 与 ECharts 均本地化，支持断网答辩。")


def add_chapter_three(doc, assets):
    heading(doc, "第3章 系统需求分析与总体设计", 1)
    heading(doc, "3.1 系统需求分析", 2)
    heading(doc, "3.1.1 功能需求", 3)
    add_body(doc, "系统包含治理管理端和公开可信通行终端。管理端负责人员生命周期、实验室组织、门禁点、区域授权、生物特征模板、通行审计、离组销毁以及保留的课程考勤业务；终端无需人员登录，在指定门禁点调用摄像头，同时识别画面中的多张人脸并给出逐人放行或拒绝反馈。")
    add_table(doc,
              ["编号", "需求", "说明"],
              [
                  ("F1", "人员生命周期", "按人员编号建档，维护类型、职责、入组/离组时间、在组/暂停状态与备用核验方式"),
                  ("F2", "实验室与门禁", "维护实验室组织、房间位置、主入口与设备区等门禁点"),
                  ("F3", "动态区域授权", "为人员授予指定门禁、日期范围与每日时段，支持立即撤销"),
                  ("F4", "多人可信通行", "每张脸独立跟踪、识别、活体判断，再执行状态、成员和授权策略"),
                  ("F5", "离组自动销毁", "撤销全部权限，物理删除档案和人脸模板，生成匿名销毁回执"),
                  ("F6", "审计与可视化", "展示识别证据、通行结果、拒绝原因、风险、时延和隐私资产清单"),
                  ("F7", "考勤兼容", "保留课程、场次、正常/迟到/缺勤判定及 UTF-8 CSV 导出"),
              ],
              [900, 2200, 5970], caption="表 3-1  系统主要功能需求",
              alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    heading(doc, "3.1.2 非功能需求", 3)
    for item in [
        "实时性：预热后单帧端到端处理应明显低于用户可感知的长时间等待，并以 P50、P95 而非只看平均值评价；",
        "安全性：陌生人不得被强制分类；活体证据不足时不得直接放行；关键管理操作必须鉴权并防止 CSRF；",
        "可解释性：系统需要输出质量、相似度、候选间隔、活体概率、风险分和端到端时延；",
        "可靠性：同一轨迹只形成一次最终决策；同一人员同一场次只形成一条考勤记录；重复请求由数据库约束和业务逻辑共同兜底；",
        "隐私性：人脸模板与个人档案分离，不保存上传原图、不下发特征；录脸需要授权记录并提供校园卡或人工核验替代方式；",
        "可销毁性：人员离组后不再保留可直接反查身份的档案与模板，销毁范围、时间和结果以匿名回执证明；",
        "可部署性：核心推理在普通 CPU 上运行，模型与前端资源可本地加载，断网条件下可完成答辩演示。",
    ]:
        add_bullet(doc, item)

    heading(doc, "3.2 系统总体架构", 2)
    add_picture(doc, assets["architecture"], "图 3-1  系统总体架构")
    add_body(doc, "系统按照交互层、接口层、算法决策层、可信访问策略层、生命周期数据层和治理展示层划分。浏览器只负责图像采集和结果展示，身份及权限判定权保留在服务端；算法层形成轨迹级身份与活体结论后，策略层继续验证人员状态、实验室成员关系、门禁点和授权有效期，再形成允许或拒绝结果。考勤服务作为通行成功后的可选联动，而不是决定门禁权限的唯一依据。")

    heading(doc, "3.3 核心数据模型", 2)
    add_picture(doc, assets["data_model"], "图 3-2  核心数据实体及关系")
    add_body(doc, "User 保存人员基础档案和生命周期状态，FaceTemplate 独立保存可撤销的人脸特征；OrganizationUnit、LabMembership 和 AccessPoint 描述实验室、成员关系与物理入口，AccessGrant 描述人员对特定门禁的日期和时段授权。RecognitionEvent 保存轨迹级身份与活体证据，AccessEvent 保存最终允许/拒绝及原因，ErasureTask 保存不含姓名、编号和人脸特征的销毁回执。该结构实现生物模板、身份档案、业务权限与审计证据的分层解耦。")
    add_table(doc,
              ["实体", "关键字段", "约束/用途"],
              [
                  ("User", "number, person_type, lifecycle_status, joined_at, expected_leave_at", "人员编号唯一；描述在组、暂停和预计离组信息"),
                  ("FaceTemplate", "user_id, engine, encoding, consented_at, active", "与档案分离的生物特征模板；每人每引擎唯一"),
                  ("OrganizationUnit", "code, name, room, is_demo", "实验室组织与物理位置"),
                  ("LabMembership", "user_id, organization_id, valid_from, valid_until", "人员与实验室归属及有效期"),
                  ("AccessPoint", "organization_id, code, name, gate_type", "实验室主入口或设备区等物理门禁点"),
                  ("AccessGrant", "user_id, access_point_id, dates, daily_window", "显式区域授权、日期与每日时段"),
                  ("AccessEvent", "decision, reason_code, access_point_id, recognition_event_id", "允许或拒绝及可解释原因"),
                  ("ErasureTask", "receipt_code, completed_at, erased_counts", "离组物理清理的匿名不可反查回执"),
              ],
              [1700, 3600, 3770], caption="表 3-2  核心实体说明",
              alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    heading(doc, "3.4 可信通行与离组业务流程", 2)
    add_picture(doc, assets["pipeline"], "图 3-3  实验室可信通行决策流程")
    add_body(doc, "公开通行终端携带门禁点编码并以固定间隔上传最新帧，服务端为每张人脸分配轨迹。身份、活体和投票通过后，策略服务依次校验门禁是否启用、人员是否在组、预计离组时间是否到期、实验室成员关系是否有效，以及是否存在覆盖当前时间的显式授权。任一条件不满足即生成拒绝 AccessEvent；全部通过才允许通行，并可联动 SignIn 与 AttendanceRecord。")
    add_body(doc, "离组流程不等同于把人员状态改为不可用。管理员输入二次确认后，系统在事务中撤销授权，删除成员关系、人脸模板、兼容特征字段、考勤关联和个人档案；历史识别、风险与通行事件清空人员外键，仅保留时间、门禁、算法指标和拒绝原因。最后生成记录各类删除数量的 ErasureTask 匿名回执。")

    heading(doc, "3.5 安全与隐私设计", 2)
    add_body(doc, "系统从接口、身份、权限、数据和版本管理五个层面保护安全与隐私。接口层限制请求体并验证特征维度；管理端使用登录、角色授权、POST 变更和 CSRF 令牌。权限层采用默认拒绝与显式授权，不因识别成功自动扩大区域访问。数据层通过唯一约束和事务保证授权、撤权及销毁的一致性，并以 is_demo 隔离真实数据与答辩演示数据。")
    add_body(doc, "威胁模型覆盖五类主要风险：陌生人被误识别、照片或屏幕重放绕过活体、合法身份访问未授权区域、离组人员的过期身份仍可使用，以及管理员越权查询或导出敏感数据。系统因此把身份、活体、在组状态、成员关系和门禁授权作为相互独立的必要条件，并对冲突身份、活体失败与权限拒绝保留结构化原因。")
    add_body(doc, "隐私设计遵循目的限定、数据最少化和存储期限约束。系统不持久化识别请求中的原始图像，将个人档案与人脸模板分表保存；录脸需记录知情授权，并提供校园卡或人工核验作为替代路径。离组销毁在同一事务内清除模板、档案、关系与授权，避免只把状态改成禁用而长期残留生物特征。")
    add_body(doc, "匿名审计信息只保留无法反推个人的事件统计、门禁点、算法指标和销毁计数；历史事件解除 person_id 外键后仍可用于安全复盘。ErasureTask 记录操作时间、任务编号与各类删除数量，使管理员能够证明清理动作已经执行，同时不在回执中复制姓名、学号或人脸特征。")
    add_body(doc, "人脸模板只保存在服务端且不保存上传原图；录入界面要求记录知情授权时间，并提供校园卡或人工核验作为替代方式。Git 忽略 data.sqlite、.env、人脸评估原图、上传目录和日志，公开评估报告使用匿名标签。当前原型已支持离组物理清理与匿名回执，正式部署仍应增加数据库加密、独立密钥管理、分级管理员权限和法定保留期限配置。")


def add_chapter_four(doc):
    heading(doc, "第4章 核心算法与决策策略设计", 1)
    heading(doc, "4.1 可解释人脸质量评分", 2)
    add_body(doc, "为避免每帧都执行特征提取，本文根据清晰度、曝光、脸部相对尺寸、近似正脸姿态和 YuNet 检测置信度构造 0～1 的质量分。清晰度使用人脸灰度图 Laplacian 方差并以 160 归一化；曝光分衡量平均亮度与 127.5 的偏差；尺寸分由人脸短边占图像短边比例得到；姿态分根据鼻尖相对双眼中点的水平偏移估计。综合质量为：")
    add_equation(doc, "Q = 0.30Q_s + 0.20Q_e + 0.20Q_z + 0.15Q_p + 0.15Q_c", "（4-1）")
    add_body(doc, "其中 Q_s、Q_e、Q_z、Q_p、Q_c 分别代表清晰度、曝光、尺寸、姿态和检测置信度。质量分只用于调度，不等同于身份置信度。当前门限为 0.38；低于门限的帧不提取 SFace 特征，但继续更新轨迹位置，以避免模糊瞬间导致轨迹和投票状态被清空。")

    heading(doc, "4.2 质量加权时序特征融合", 2)
    add_body(doc, "浏览器视频中单帧特征会随模糊、表情和轻微姿态变化而波动。本文在每条轨迹内保存一个融合特征。设当前归一化特征为 f_t，上一时刻融合特征为 F_(t-1)，当前质量为 Q_t，则融合权重为：")
    add_equation(doc, "α_t = clip(0.35 + 0.50Q_t, 0.45, 0.85)", "（4-2）")
    add_equation(doc, "F_t = norm((1-α_t)F_(t-1) + α_t f_t)", "（4-3）")
    add_body(doc, "高质量帧对融合结果具有更大影响，最低可接受帧仍保留 0.45 的权重，以便轨迹误关联恢复后快速修正。融合特征以 float16 和 Base64 形式保存在签名会话中，每条轨迹约占 344 B，兼顾会话大小与数值稳定性。")

    heading(doc, "4.3 质量约束的开放集双阈值", 2)
    add_body(doc, "系统使用最高余弦相似度和候选间隔联合判定身份。基础相似度阈值为 0.40，最小候选间隔为 0.08。为了保证质量自适应不会扩大误识风险，低质量帧只允许收紧阈值：")
    add_equation(doc, "τ(Q) = τ_0 + max(0, 0.60-Q) × 0.10", "（4-4）")
    add_body(doc, "只有当 s_(1) ≥ τ(Q) 且 s_(1)-s_(2) ≥ 0.08 时，候选身份才被接受。该策略不会因画面变差而降低门限。身份结果还必须在 6 s 窗口内累计 3 票，单帧误识不能直接形成通行请求。")

    heading(doc, "4.4 风险自适应活体检测", 2)
    add_body(doc, "MiniFASNetV2 对每条尚未完成的 SFace 轨迹输出真人概率 P_live。为降低单帧活体波动，系统采用质量加权指数移动平均，权重为 clip(0.35+0.40Q, 0.40, 0.75)。综合风险分由静默活体、身份强度、图像质量和证据稳定度组成：")
    add_equation(doc, "R = 0.55(1-P_live) + 0.20R_id + 0.15(1-Q) + 0.10(1-S)", "（4-5）")
    add_body(doc, "其中 R_id 表示身份置信度低于 0.55 的归一化风险，S 表示已累计活体样本数相对最小帧数 2 的稳定度。累计至少 2 帧后，若 P_live≥0.80、相似度≥0.55 且质量≥0.55，则静默放行；若 P_live≤0.15，则直接阻断；其余情况进入中风险状态，随机要求眨眼或转头一次。若模型缺失或推理失败，系统安全降级为单动作挑战，而不是直接放行。")

    heading(doc, "4.5 动作挑战状态机", 2)
    add_body(doc, "眨眼检测使用 EAR。系统先观察 EAR≥0.24 的睁眼状态，再观察 EAR≤0.20 的闭眼状态，随后重新达到睁眼阈值时记为完成眨眼。转头使用鼻尖相对双眼中点的归一化水平位置，若挑战开始后最大值与最小值之差达到 0.12，则记为完成转头。随机指令发出时会清空此前动作状态，降低预录视频恰好包含动作而通过的概率。动作窗口为 8 s，超时重新开始。")

    heading(doc, "4.6 多人身份冲突与最终决策", 2)
    add_body(doc, "同一帧中每条轨迹独立完成识别。但如果两张脸同时匹配到同一个注册身份，系统无法可靠判断哪一条是真人、哪一条可能是照片或屏幕，因此将该身份加入冲突集合：相关轨迹的投票清零、活体状态重置、两者均不得通行，并记录高风险 identity_collision 事件。该规则以牺牲冲突场景下的通过率换取更低的误放行风险。")
    add_body(doc, "身份票数达到要求且活体通过，只代表形成可信身份候选。每条轨迹仅在成功、重复、风险拒绝或权限拒绝等最终状态形成时写入 RecognitionEvent；高风险结论再写入 RiskEvent。该设计避免逐帧日志膨胀，同时保留质量、相似度、候选间隔、活体概率、风险分和时延证据。")

    heading(doc, "4.7 最小权限通行决策", 2)
    add_body(doc, "设身份与活体条件为 I，人员在组状态为 S，有效实验室成员关系为 M，门禁点显式授权及有效时段为 G，则带门禁参数的通行条件为：")
    add_equation(doc, "Allow = I ∧ S ∧ M ∧ G", "（4-6）")
    add_body(doc, "系统采用短路拒绝并返回结构化 reason_code。门禁不存在或停用返回 gate_not_found/gate_disabled；人员暂停或到期返回 person_inactive/person_expired；不属于该实验室返回 membership_missing；缺少授权、授权未开始、已过期或不在每日时段内分别返回 grant_missing、grant_not_started、grant_expired 与 outside_daily_window。该顺序使管理员能够解释拒绝原因，同时避免识别成功被误解为拥有任意区域权限。")

    heading(doc, "4.8\u3000新旧特征空间兼容", 2)
    add_body(doc, "历史系统使用 dlib HOG 检测与 ResNet 128 维特征。SFace 与 dlib 均为 128 维，但属于不同特征空间，不能直接混合比较。因此 User 表分别保存 face_encoding 与 face_encoding_sface。auto 模式只在数据库存在 SFace 特征时启用高速链路；否则回退 dlib。管理员可通过重新上传注册照片或 enroll-sface 命令为历史人员生成 SFace 特征，并使用旧 dlib 特征校验照片身份一致性。")


def add_chapter_five(doc):
    heading(doc, "第5章 系统实现", 1)
    heading(doc, "5.1 开发与运行环境", 2)
    add_table(doc,
              ["类别", "配置"],
              [
                  ("语言与运行时", "Python 3.11.15"),
                  ("Web 框架", "Flask 3.0.3、Jinja2、Werkzeug"),
                  ("数据层", "SQLAlchemy 2.0.32、SQLite、Flask-Migrate"),
                  ("计算机视觉", "OpenCV 4.10.0、dlib 19.24.6、NumPy 1.26.4"),
                  ("前端可视化", "Bootstrap 5、本地 Apache ECharts 6.1"),
                  ("测试", "pytest；共 59 项自动化测试"),
              ],
              [2200, 6870], caption="表 5-1  主要开发环境与依赖",
              alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    heading(doc, "5.2 Web 可信通行接口实现", 2)
    add_body(doc, "通行页通过 getUserMedia 获取视频流，Canvas 将最新画面压缩为 JPEG，并以 image/jpeg 二进制请求 POST /api/sign-in-image?gate=<门禁编码>。服务端限制请求体最大为 8 MiB，兼容旧版 Base64 data URL，但新页面优先使用二进制以减少网络与解析开销。接口返回 multi 状态，其中 faces 数组包含每张脸的边界框、轨迹编号、质量、相似度、票数、活体与权限状态，signed 数组包含本帧最终形成的允许或拒绝决定。")
    add_body(doc, "前端人脸框按轨迹编号稳定复用，不随每次检测结果重新创建，从而避免视觉闪烁。每个轨迹独立显示“改善光照”“保持正对摄像头”“请眨眼”“允许通行”或具体拒绝原因；处理请求采用最新帧优先策略，避免网络排队导致界面反应滞后。")

    heading(doc, "5.3 人员生命周期与动态授权实现", 2)
    add_body(doc, "User 增加 person_type、lifecycle_status、joined_at、expected_leave_at、suspended_at、offboarded_at、privacy_consent_at 和 alternative_auth 字段；FaceTemplate 将人脸特征从档案语义中独立出来。管理员可先建档后录脸，并在个人详情页查看实验室归属、模板状态、区域授权与生命周期事件。OrganizationUnit 和 AccessPoint 支持维护实验室及多个物理入口，AccessGrant 将授权精确到人员、门禁点、日期范围和每日时段。")
    add_body(doc, "evaluate_access 以默认拒绝方式实现策略判断。暂停操作不删除档案，但立即使全部门禁决策失败；恢复后仍需原授权有效才可进入。offboard_and_erase 在单个数据库事务中统计并删除模板、授权、成员关系和个人记录，同时把历史审计事件的 person_id 置空，最后创建只含随机回执编号、删除数量与完成时间的 ErasureTask。")

    heading(doc, "5.4 课程与考勤兼容实现", 2)
    add_body(doc, "管理员可创建课程和独立考勤场次，并设置开始时间、迟到阈值和结束时间。同一数据空间只允许一个活动场次。识别成功后，系统先写入每日 SignIn 兼容表，再检查活动场次和选课关系；在迟到阈值前写入 on_time，之后写入 late。场次关闭时，为尚未产生记录的选课学生批量生成 absent。手工补签使用 manual 状态并保留操作方式。")
    add_body(doc, "场次详情调用统一的 build_session_detail 聚合函数，页面和 CSV 导出共用同一逐学生状态列表。活动场次中尚未签到的学生显示 pending，关闭场次中的缺失记录显示 absent；缺失的识别指标使用“--”而不是用 0 填充，避免把没有数据误解为低风险或低时延。")

    heading(doc, "5.5 可视化、隐私与实验中心实现", 2)
    add_body(doc, "LabGuard 驾驶舱在原有出勤、趋势、热力图、识别决策和时延指标之上，增加当前在组/暂停/即将到期人数、实验室数量、区域通行决策和匿名销毁凭证。人员生命周期页、实验室与门禁页、通行决策页、人员完整档案和隐私销毁中心使用统一浅色视觉系统，便于答辩展示。演示数据通过固定随机种子生成，明确标记 is_demo，不创建任何人脸特征，仅用于展示业务界面。")
    add_body(doc, "算法实验中心直接读取 docs/performance_baseline.md 与 evaluation_report/sface_metrics.json，不在模板中硬编码实验结论。页面同时显示实验来源时间和证据完成度矩阵，将性能对比标为完成，将小样本阈值实验标为受限，将多人并发、呈现攻击和消融实验标为待完成。这种设计避免用演示数据填补真实实验空缺。")

    heading(doc, "5.6 异常处理与兼容设计", 2)
    add_body(doc, "当 YuNet/SFace 模型缺失且配置为 opencv 强制模式时，接口返回 503 和明确模型提示；auto 模式下则回退 dlib。MiniFASNetV2 缺失或推理失败时，活体策略回退为单次动作挑战。特征解析会拒绝非 128 维、非有限值或零范数向量。RecognitionEvent 的数值字段在落库前统一转为有限浮点数，否则保存为空值，避免 NaN/Infinity 破坏 JSON 与统计。")

    heading(doc, "5.7 测试实现", 2)
    add_body(doc, "项目建立 pytest 测试集，覆盖密码哈希与旧密码升级、管理员权限、CSRF、防重复签到、日期统计、SFace 归一化与开放集匹配、质量评分、多人轨迹关联、特征融合、活体状态机、照片攻击阻断、门禁显式授权、人员暂停、离组物理销毁、考勤场次业务、风险事件、实验页和 CSV 导出等。测试使用内存 SQLite 和隔离应用上下文，不依赖本机真实人员数据库。")
    add_table(doc,
              ["测试类别", "代表性验证点"],
              [
                  ("安全与权限", "密码哈希、旧明文升级、普通学生拒绝访问管理页、CSRF"),
                  ("识别算法", "特征归一化、余弦匹配、候选间隔、质量评分、异常向量"),
                  ("多人和时序", "轨迹分配、低质量保持、融合向量、身份冲突、投票"),
                  ("活体检测", "静默通过、低分阻断、中风险动作挑战、模型缺失降级"),
                  ("考勤业务", "激活/关闭场次、迟到判定、缺勤补齐、明细与 CSV 对账"),
                  ("生命周期治理", "显式区域授权、暂停阻断、离组撤权、模板物理删除与匿名审计"),
                  ("页面与数据", "驾驶舱、识别中心、实验中心、风险页及 API 可访问性"),
              ],
              [2300, 6770], caption="表 5-2  自动化测试覆盖范围",
              alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
              note="说明：当前共 59 项测试全部通过；测试通过证明实现符合既定逻辑，不等同于真实场景算法准确率。")


def add_chapter_six(doc, assets, metrics):
    heading(doc, "第6章 系统测试与实验分析", 1)
    heading(doc, "6.1 实验原则与指标口径", 2)
    add_body(doc, "本章严格区分工程基准、功能性小样本验证、演示数据和待补正式实验。工程基准用于比较同一计算机、同一图像输入下两条流水线的耗时；小样本验证只检查相似度方向、阈值代码和曲线输出是否合理；演示数据只用于前端展示；多人识别率、攻击检测 ACER 和时序融合收益必须等待授权数据后测量。")
    add_table(doc,
              ["证据", "当前状态", "可支持结论", "不能支持结论"],
              [
                  ("45 个流水线样本", "已完成", "同机同输入下的耗时对比", "不同硬件、多人视频时延"),
                  ("1 身份/2 正/11 负", "受限", "阈值流程和分数方向正确", "多人 Rank-1、普适 FAR/FRR"),
                  ("59 项自动化测试", "已完成", "软件逻辑、权限和销毁回归稳定性", "模型真实场景准确率"),
                  ("答辩演示数据", "仅展示", "生命周期、门禁、考勤与统计闭环", "任何算法准确率或攻击阻断率"),
                  ("多人/攻击/消融", "待采集", "—", "正式论文结论待补"),
              ],
              [2100, 1250, 2750, 2970], caption="表 6-1  当前证据边界",
              alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER,
                          WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    heading(doc, "6.2 人脸流水线性能对比", 2)
    add_body(doc, "测试平台为 Intel Core i7-12650H、Windows 11、Python 3.11.15 和 OpenCV 4.10.0。输入为 15 张人脸图像，最长边统一缩放至 640 像素，每张图像运行 3 次，共 45 个样本。比较 dlib HOG 检测加 dlib ResNet 编码与 YuNet 检测加 SFace 编码。")
    add_table(doc,
              ["流水线", "中位耗时/ms", "P95/ms", "平均耗时/ms", "检出人脸数"],
              [
                  ("dlib HOG + dlib ResNet", "286.61", "390.25", "299.18", "42"),
                  ("YuNet + SFace", "16.65", "32.85", "19.97", "45"),
              ],
              [3000, 1500, 1300, 1600, 1670], caption="表 6-2  两条人脸流水线性能对比",
              note="样本量为 45；该表比较检测与特征提取流水线，不包含多人跟踪、网络传输、活体状态累计和数据库写入。")
    add_picture(doc, assets["performance"], "图 6-1  两条人脸流水线耗时对比")
    add_body(doc, "YuNet 与 SFace 的中位耗时相对 dlib 流水线加速 17.22 倍，P95 从 390.25 ms 降至 32.85 ms。结果说明替换检测与特征提取链路能够显著降低 CPU 推理时延，是改善“眨眼后等待很久”问题的主要基础。检出人脸数存在 42 与 45 的差异，提示两条检测器对部分样本的敏感性不同，因此不能只比较速度，正式实验还需要报告多人视频中的漏检率。")

    heading(doc, "6.3 Web 端到端工程基准", 2)
    add_body(doc, "使用一张 480×640、约 74 KiB 的 JPEG 通过 Flask 测试客户端连续请求。模型冷启动首帧约为 174～202 ms；预热后完整请求约为 42～47 ms；MiniFASNetV2 静默活体推理约为 1.4～1.6 ms/脸；高置信真人需要累计 2 帧静默证据后放行。由于浏览器侧还存在摄像头帧率、JPEG 编码和网络调度，本结果只作为本机工程基准。")
    add_callout(doc, "阶段性结论", "预热后单脸完整请求已处于几十毫秒量级，交互等待主要由多帧证据累计而非单次神经网络推理决定。后续优化应优先减少无效帧、采用最新帧优先并验证多人数量增长对 P95 的影响。", color=BLUE)

    heading(doc, "6.4 SFace 小样本阈值验证", 2)
    registered = metrics["dataset"]["registered_identities"]
    positives = metrics["dataset"]["positive_samples"]
    negatives = metrics["dataset"]["negative_samples"]
    threshold = metrics["threshold"]
    add_body(doc, f"现有数据集包含 {registered} 个注册身份、{positives} 个正样本和 {negatives} 个负样本。使用余弦相似度阈值 {threshold:.2f} 时，当前样本上 FAR 与 FRR 均为 0，Rank-1 为 100%。但正样本仅 2 个且只有 1 个注册身份，Rank-1 事实上没有多人分类难度，错误率也具有极大统计不确定性，因此本结果只证明代码能够区分当前样本，不作为正式准确率结论。")
    add_picture(doc, assets["score_plot"], "图 6-2  小样本 SFace 余弦相似度分布")
    add_body(doc, "图 6-2 中正样本余弦相似度为 0.589 和 0.805，负样本最高为 0.120，当前阈值 0.40 位于两类之间。该间隔可能受到采集设备、姿态和样本选择影响。正式阈值应在独立验证集上确定，并以至少 10 个身份、每人多种光照与角度样本以及独立陌生人集重新计算 FAR、FRR、ROC 和置信区间。")

    heading(doc, "6.5 自动化测试结果", 2)
    add_body(doc, "运行 pytest 完整测试集，59 项测试全部通过。测试覆盖输入校验、权限安全、特征匹配、轨迹融合、活体状态、门禁授权、暂停阻断、离组销毁、业务落库、统计页面和 CSV 导出。浏览器验收中，LabGuard 驾驶舱、28 名匿名演示成员列表、人员完整档案、实验室与门禁、通行决策、隐私销毁中心和带门禁参数的通行终端均正确渲染，浏览器控制台未发现错误。")

    heading(doc, "6.6 待补正式实验设计", 2)
    heading(doc, "6.6.1 多人并发时延实验", 3)
    add_body(doc, "人数设置为 1、2、4、6 人，每档至少采集 3 段 30 s 视频，固定分辨率、采样间隔与摄像头位置；同一视频预热后重复运行 3 次。记录实际出现人脸数、检测人脸数、成功身份数、平均时延、P50、P95、漏检率、轨迹切换次数和身份抖动次数。")
    add_table(doc,
              ["人数", "视频数", "检测召回率/%", "最终识别率/%", "P50/ms", "P95/ms", "轨迹切换"],
              [("1", "≥3", "待测", "待测", "待测", "待测", "待测"),
               ("2", "≥3", "待测", "待测", "待测", "待测", "待测"),
               ("4", "≥3", "待测", "待测", "待测", "待测", "待测"),
               ("6", "≥3", "待测", "待测", "待测", "待测", "待测")],
              [850, 1000, 1500, 1500, 1250, 1250, 1720], caption="表 6-3  多人并发实验结果占位表",
              note="判定建议：重点检验人数增加后端到端 P95 是否仍小于 200 ms，以及陌生人是否阻塞已注册人员。")

    heading(doc, "6.6.2 照片与视频重放攻击实验", 3)
    add_body(doc, "分别采集真人、打印照片、手机静态照片和手机视频重放，每类至少 30 次独立尝试。记录是否静默通过、是否触发动作、最终是否放行以及总耗时。按 ISO/IEC 30107-3 相关口径计算攻击呈现分类错误率 APCER、真人呈现分类错误率 BPCER 和 ACER[13]：")
    add_equation(doc, "ACER = (APCER + BPCER) / 2", "（6-1）")
    add_table(doc,
              ["样本类型", "次数", "静默通过", "触发挑战", "最终放行", "错误率/%"],
              [("真人", "≥30", "待测", "待测", "待测", "BPCER 待测"),
               ("打印照片", "≥30", "待测", "待测", "待测", "APCER 待测"),
               ("手机静态照片", "≥30", "待测", "待测", "待测", "APCER 待测"),
               ("手机视频重放", "≥30", "待测", "待测", "待测", "APCER 待测")],
              [1800, 900, 1500, 1500, 1450, 1920], caption="表 6-4  呈现攻击实验结果占位表")

    heading(doc, "6.6.3 时序特征融合消融实验", 3)
    add_body(doc, "使用完全相同的视频输入比较三种方案：A 为单帧 SFace；B 为多帧投票但不融合特征；C 为质量加权时序特征融合的完整系统。指标包括每条轨迹相似度标准差、身份抖动次数、最终识别率、平均确认帧数和 P95 时延。只有输入视频一致，才能把差异归因于融合策略。")
    add_table(doc,
              ["方案", "相似度标准差", "身份抖动/次", "最终识别率/%", "确认帧数", "P95/ms"],
              [("A 单帧 SFace", "待测", "待测", "待测", "1", "待测"),
               ("B 多帧投票", "待测", "待测", "待测", "待测", "待测"),
               ("C 质量加权融合", "待测", "待测", "待测", "待测", "待测")],
              [1900, 1700, 1500, 1750, 1100, 1120], caption="表 6-5  时序融合消融实验占位表")

    heading(doc, "6.7 有效性威胁与限制", 2)
    for item in [
        "样本规模限制：当前身份与攻击样本不足，不能估计可推广的 FAR、FRR、APCER、BPCER 或 ACER；",
        "设备域差异：RGB 静默活体对摄像头、显示屏、打印设备和光照敏感，本机阈值不能直接迁移；",
        "多人轨迹限制：轻量 IoU/中心距离关联未使用运动预测或外观重识别，遮挡和快速交叉可能导致轨迹切换；",
        "统计独立性：连续视频帧高度相关，正式实验应以独立尝试或独立视频为统计单位，而非把每帧当成独立样本；",
        "演示数据边界：28 名匿名成员、实验室、门禁授权与历史场次均为确定性生成的业务演示数据，只验证页面与数据闭环；",
        "部署边界：系统当前面向局域网或单机演示，生产部署仍需 HTTPS、数据库加密、日志脱敏和更严格的授权策略。",
    ]:
        add_bullet(doc, item)


def add_chapter_seven(doc):
    heading(doc, "第7章 总结与展望", 1)
    heading(doc, "7.1 工作总结", 2)
    add_body(doc, "本文完成了一套基于深度学习的高校实验室人员全生命周期可信通行平台。系统使用 YuNet、SFace 和 MiniFASNetV2 构建 CPU 实时推理链路，通过质量评分、多人轨迹、时序特征融合、开放集双阈值、多帧投票和风险自适应活体，将单帧模型输出转化为稳定的轨迹级身份结论。业务层进一步建立人员、实验室、成员关系、门禁点、动态授权、通行事件和匿名销毁回执之间的关系，实现“入组建档—最小授权—可信通行—暂停阻断—离组销毁”的治理闭环，同时保留课程考勤联动能力。")
    add_body(doc, "已有基准显示 YuNet 与 SFace 流水线的中位耗时为 16.65 ms，相比 dlib 基线加速 17.22 倍；预热后 Web 请求约 42～47 ms；59 项自动化测试全部通过。更重要的是，系统明确区分真实实验与演示数据，没有使用模拟业务数据冒充算法准确率，也没有把识别成功直接等同于区域放行。")

    heading(doc, "7.2 后续工作", 2)
    add_body(doc, "下一阶段将首先完成授权数据采集，按第6章方案补齐多人并发、照片/视频攻击和时序融合消融实验，并为 FAR、FRR、APCER、BPCER 和 P95 时延给出样本量与置信区间。其次，可引入带运动预测的轨迹管理、服务器端 Redis 状态或 WebSocket 流式传输。最后，可研究加密或可撤销生物模板、审批流、到期定时任务、分级管理员和真实门禁控制器协议，使原型进一步接近高校实验室部署要求。")
    add_callout(doc, "答辩口径建议", "核心创新可概括为：不是再做一个人脸签到页面，而是把高速多人识别作为可信身份入口，进一步实现实验室人员全生命周期、区域最小权限与离组可验证销毁。速度结论可展示，准确率与攻击结论等待真实数据后再定稿。", color=GOLD, fill="FFF7E8")


def add_references(doc):
    heading(doc, "参考文献", 1)
    refs = [
        "TURK M, PENTLAND A. Eigenfaces for recognition[J]. Journal of Cognitive Neuroscience, 1991, 3(1): 71-86. DOI: 10.1162/jocn.1991.3.1.71.",
        "VIOLA P, JONES M. Rapid object detection using a boosted cascade of simple features[C]//Proceedings of CVPR. 2001: I-511-I-518. DOI: 10.1109/CVPR.2001.990517.",
        "ZHANG K, ZHANG Z, LI Z, et al. Joint face detection and alignment using multitask cascaded convolutional networks[J]. IEEE Signal Processing Letters, 2016, 23(10): 1499-1503. DOI: 10.1109/LSP.2016.2603342.",
        "SCHROFF F, KALENICHENKO D, PHILBIN J. FaceNet: A unified embedding for face recognition and clustering[C]//Proceedings of CVPR. 2015: 815-823.",
        "DENG J, GUO J, XUE N, et al. ArcFace: Additive angular margin loss for deep face recognition[C]//Proceedings of CVPR. 2019: 4690-4699.",
        "ZHONG Y, DENG W, HU J, et al. SFace: Sigmoid-constrained hypersphere loss for robust face recognition[J]. IEEE Transactions on Image Processing, 2021, 30: 2587-2598. DOI: 10.1109/TIP.2020.3048632.",
        "WU W, PENG H, YU S. YuNet: A tiny millisecond-level face detector[J]. Machine Intelligence Research, 2023, 20: 656-665. DOI: 10.1007/s11633-023-1423-y.",
        "SCHEIRER W J, ROCHA A, SAPKOTA A, et al. Toward open set recognition[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013, 35(7): 1757-1772. DOI: 10.1109/TPAMI.2012.256.",
        "BEWLEY A, GE Z, OTT L, et al. Simple online and realtime tracking[C]//Proceedings of ICIP. 2016: 3464-3468. DOI: 10.1109/ICIP.2016.7533003.",
        "BOULKENAFET Z, KOMULAINEN J, LI L, et al. OULU-NPU: A mobile face presentation attack database with real-world variations[C]//Proceedings of FG. 2017: 612-618. DOI: 10.1109/FG.2017.77.",
        "MINIVISION-AI. Silent-Face-Anti-Spoofing[EB/OL]. GitHub, 2020[2026-08-24]. https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.",
        "SOUKUPOVÁ T, ČECH J. Real-time eye blink detection using facial landmarks[C]//21st Computer Vision Winter Workshop. 2016.",
        "JOURABLOO A, LIU Y, LIU X. Face de-spoofing: Anti-spoofing via noise modeling[C]//Proceedings of ECCV. 2018: 290-306.",
        "ISO/IEC. ISO/IEC 30107-3:2017 Information technology—Biometric presentation attack detection—Part 3: Testing and reporting[S]. Geneva: ISO, 2017.",
        "KING D E. Dlib-ml: A machine learning toolkit[J]. Journal of Machine Learning Research, 2009, 10(60): 1755-1758.",
        "HUANG G B, RAMESH M, BERG T, et al. Labeled faces in the wild: A database for studying face recognition in unconstrained environments[R]. Amherst: University of Massachusetts, 2007.",
        "OPENCV. DNN-based face detection and recognition[EB/OL].[2026-08-24]. https://docs.opencv.org/master/d0/dd4/tutorial_dnn_face.html.",
        "PALLETS PROJECTS. Flask documentation[EB/OL].[2026-08-24]. https://flask.palletsprojects.com/.",
        "SQLALCHEMY AUTHORS AND CONTRIBUTORS. SQLAlchemy documentation[EB/OL].[2026-08-24]. https://docs.sqlalchemy.org/.",
    ]
    for index, ref in enumerate(refs, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Pt(31)
        p.paragraph_format.hanging_indent = Pt(31)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(f"[{index}]  {ref}")
        set_font(run, size=10.5, color=INK)


def add_acknowledgements_and_appendix(doc):
    heading(doc, "致谢", 1)
    add_body(doc, "[待填写：感谢指导教师在选题、系统设计与论文修改方面的指导；感谢参与后续授权数据采集与测试的同学；感谢家人和朋友在毕业设计期间的支持。正式提交前请根据学校规范补充并个性化修改。]")

    heading(doc, "附录A 关键配置参数", 1)
    add_table(doc,
              ["参数", "当前值", "作用"],
              [
                  ("FACE_ENGINE", "auto", "存在 SFace 特征时使用高速链路，否则回退 dlib"),
                  ("FACE_MIN_QUALITY", "0.38", "低于该值不提取身份特征"),
                  ("SFACE_COSINE_THRESHOLD", "0.40", "SFace 基础相似度阈值"),
                  ("SFACE_MINIMUM_MARGIN", "0.08", "第一与第二候选最小间隔"),
                  ("FACE_MATCH_VOTES", "3", "最终落库前连续身份票数"),
                  ("FACE_MATCH_WINDOW", "6.0 s", "身份票数有效时间窗口"),
                  ("FACE_TRACK_TTL", "2.0 s", "漏检或离场后的轨迹保留时间"),
                  ("LIVENESS_PASSIVE_MIN_FRAMES", "2", "静默活体最少累计帧数"),
                  ("LIVENESS_PASSIVE_ACCEPT", "0.80", "高置信真人阈值"),
                  ("LIVENESS_PASSIVE_REJECT", "0.15", "疑似攻击拒绝阈值"),
                  ("LIVENESS_WINDOW", "8.0 s", "随机动作挑战时间窗口"),
              ],
              [3100, 1600, 4370], caption="表 A-1  系统主要识别参数",
              alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    heading(doc, "附录B 后续数据补全清单", 1)
    for item in [
        "收集不少于 10 个授权身份，每个身份至少 5 张不同光照、角度和距离的照片，并单独保留陌生人集；",
        "采集 1、2、4、6 人视频，每档不少于 3 段、每段 30 s，同一视频重复运行 3 次；",
        "采集真人、打印照片、手机静态照片、手机视频重放各不少于 30 次独立尝试；",
        "固定同一批视频完成单帧、多帧投票、质量加权时序融合三组消融；",
        "保存原始实验清单、参数、时间戳和脚本输出，更新第6章表 6-3 至表 6-5；",
        "正式提交前获取学校 Word 模板，替换封面字段，核对页眉、页码、目录、图表编号和参考文献格式。",
    ]:
        add_bullet(doc, item)


def main():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assets = build_assets(metrics)
    doc = Document()
    configure_styles(doc)
    configure_page(doc)
    create_numbering(doc)
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "本科毕业设计论文初稿"
    doc.core_properties.creator = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.keywords = "人脸识别, 高校实验室, 人员全生命周期, 可信通行, 活体检测, 动态权限, 可验证销毁"
    doc.core_properties.comments = "初稿；已同步实验室生命周期平台，待学校模板与真实多人/攻击/消融实验数据补充"

    add_cover(doc)
    add_abstracts(doc)
    add_toc_placeholder(doc)
    add_chapter_one(doc)
    add_chapter_two(doc)
    add_chapter_three(doc, assets)
    add_chapter_four(doc)
    add_chapter_five(doc)
    add_chapter_six(doc, assets, metrics)
    add_chapter_seven(doc)
    add_references(doc)
    add_acknowledgements_and_appendix(doc)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
