"""Render the thesis PDF to page PNGs and 2x2 visual-QA sheets."""
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "docs" / "thesis_render" / "毕业设计论文初稿.pdf"
PAGE_DIR = ROOT / "docs" / "thesis_render" / "pages"
SHEET_DIR = ROOT / "docs" / "thesis_render" / "sheets"


def label_font(size=26):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main():
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF))
    pages = []
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=1.7)
        image = bitmap.to_pil().convert("RGB")
        out = PAGE_DIR / f"page-{index + 1:03d}.png"
        image.save(out, quality=95)
        pages.append((index + 1, image))

    font = label_font()
    gap = 26
    label_height = 44
    page_w = max(image.width for _, image in pages)
    page_h = max(image.height for _, image in pages)
    slot_w = page_w
    slot_h = page_h + label_height
    for sheet_index in range(0, len(pages), 4):
        batch = pages[sheet_index:sheet_index + 4]
        sheet = Image.new(
            "RGB",
            (2 * slot_w + 3 * gap, 2 * slot_h + 3 * gap),
            "#D7DEE8",
        )
        draw = ImageDraw.Draw(sheet)
        for slot, (page_number, image) in enumerate(batch):
            row, col = divmod(slot, 2)
            x = gap + col * (slot_w + gap)
            y = gap + row * (slot_h + gap)
            draw.text((x, y + 5), f"第 {page_number} 页", font=font, fill="#24384E")
            sheet.paste(image, (x, y + label_height))
        out = SHEET_DIR / f"sheet-{sheet_index // 4 + 1:02d}.png"
        sheet.save(out, quality=92)

    print(f"pages={len(pages)} sheets={(len(pages) + 3) // 4}")


if __name__ == "__main__":
    main()
