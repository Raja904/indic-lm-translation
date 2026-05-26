import os
import re
import base64
import markdown
from pathlib import Path
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Nirmala UI for Devanagari support (local copies)
NIRMALA_REGULAR = str(Path("../fonts/Nirmala.ttf").resolve())
NIRMALA_BOLD    = str(Path("../fonts/NirmalaB.ttf").resolve())

try:
    pdfmetrics.registerFont(TTFont('Nirmala', NIRMALA_REGULAR))
    pdfmetrics.registerFont(TTFont('NirmalaBold', NIRMALA_BOLD))
except Exception as e:
    print(f"Warning: Could not register Nirmala fonts: {e}")

BASE_DIR = Path("..").resolve()  # repo root

def embed_image(path_str, report_dir):
    candidates = [
        report_dir / path_str,
        BASE_DIR / path_str,
    ]
    for p in candidates:
        p = p.resolve()
        if p.exists():
            ext = p.suffix.lower().lstrip('.')
            if ext == 'jpg': ext = 'jpeg'
            with open(p, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/{ext};base64,{b64}"
    return path_str

def process_markdown(content, report_dir):
    def replace_img(match):
        alt = match.group(1)
        src = match.group(2)
        if src.startswith('data:') or src.startswith('http'):
            return match.group(0)
        embedded = embed_image(src, report_dir)
        return f"![{alt}]({embedded})"
    
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, content)

def md_to_pdf(md_path, output_path):
    print(f"Reading {md_path}...")
    content = md_path.read_text(encoding='utf-8')

    content = process_markdown(content, md_path.parent)

    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
    html_body = md.convert(content)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @page {{ margin: 2cm 2.5cm; }}
  body {{
    font-family: Nirmala, 'Segoe UI', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a1a;
  }}
  h1 {{ font-size: 20pt; color: #0d3349; border-bottom: 3px solid #0d3349; padding-bottom: 8px; margin-top: 0; }}
  h2 {{ font-size: 15pt; color: #1a5276; border-bottom: 1.5px solid #aed6f1; padding-bottom: 4px; margin-top: 28px; }}
  h3 {{ font-size: 12pt; color: #21618c; margin-top: 20px; }}
  p {{ margin: 8px 0; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 16px auto; border: 1px solid #d5d8dc; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 9.5pt; }}
  th {{ background-color: #1a5276; color: white; padding: 7px 10px; text-align: left; border: 1px solid #1a5276; }}
  td {{ padding: 6px 10px; border: 1px solid #d5d8dc; }}
  tr:nth-child(even) td {{ background-color: #eaf4fb; }}
  code {{ background: #f4f6f7; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 9pt; color: #c0392b; }}
  pre {{ background: #f4f6f7; border: 1px solid #d5d8dc; border-left: 4px solid #2e86c1; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 8.5pt; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    def link_callback(uri, rel):
        for font_path in [NIRMALA_REGULAR, NIRMALA_BOLD]:
            if os.path.basename(font_path) in uri:
                return str(Path(font_path).resolve())
        return uri

    with open(str(output_path), "wb") as pdf_file:
        result = pisa.CreatePDF(html, dest=pdf_file, encoding='utf-8', link_callback=link_callback)
    
    if result.err:
        print(f"PDF error: {result.err}")
    else:
        print(f"DONE! Saved to: {output_path}")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)  # Set working directory to scripts/
    
    reports = [
        (Path("../report/part2_final_report.md"), Path("../report/Part2_Technical_Report.pdf")),
        (Path("../report/part2_experiment_details.md"), Path("../report/Part2_Experiment_Details.pdf"))
    ]
    
    for md_path, pdf_path in reports:
        md_to_pdf(md_path, pdf_path)
