"""
Converts final_report.md to a PDF, resolving all relative image paths
so that the graphs are correctly embedded in the output PDF.
Uses xhtml2pdf (pure Python, no GTK needed on Windows).
"""

import os
import re
import base64
import markdown
from pathlib import Path
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Nirmala UI for Devanagari support (local copies)
NIRMALA_REGULAR = str(Path("fonts/Nirmala.ttf").resolve())
NIRMALA_BOLD    = str(Path("fonts/NirmalaB.ttf").resolve())
pdfmetrics.registerFont(TTFont('Nirmala', NIRMALA_REGULAR))
pdfmetrics.registerFont(TTFont('NirmalaBold', NIRMALA_BOLD))

REPORT_MD = Path("report/final_report.md")
OUTPUT_PDF = Path("report/Rajeev_Kandpal_Technical_Report.pdf")
REPORT_DIR = REPORT_MD.parent
BASE_DIR = Path(".")  # repo root

def embed_image(path_str, base_dir):
    """Resolve an image path relative to the report directory and return a data URI."""
    # Try relative to report dir first, then relative to repo root
    candidates = [
        REPORT_DIR / path_str,
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
    print(f"  WARNING: Could not find image: {path_str}")
    return path_str

def process_markdown(content, base_dir):
    """Replace all relative image paths with embedded base64 data URIs."""
    def replace_img(match):
        alt = match.group(1)
        src = match.group(2)
        # Skip already-embedded or absolute URLs
        if src.startswith('data:') or src.startswith('http'):
            return match.group(0)
        embedded = embed_image(src, base_dir)
        return f"![{alt}]({embedded})"
    
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, content)

def md_to_pdf(md_path, output_path):
    print(f"Reading {md_path}...")
    content = md_path.read_text(encoding='utf-8')

    print("Embedding images...")
    content = process_markdown(content, BASE_DIR)

    print("Converting markdown to HTML...")
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
    html_body = md.convert(content)

    # Build full HTML with professional styling
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @page {{
    margin: 2cm 2.5cm;
  }}
  body {{
    font-family: Nirmala, 'Segoe UI', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a1a;
    max-width: 100%;
  }}
  h1 {{
    font-size: 20pt;
    color: #0d3349;
    border-bottom: 3px solid #0d3349;
    padding-bottom: 8px;
    margin-top: 0;
  }}
  h2 {{
    font-size: 15pt;
    color: #1a5276;
    border-bottom: 1.5px solid #aed6f1;
    padding-bottom: 4px;
    margin-top: 28px;
  }}
  h3 {{
    font-size: 12pt;
    color: #21618c;
    margin-top: 20px;
  }}
  h4 {{
    font-size: 10.5pt;
    color: #2e86c1;
    margin-top: 16px;
  }}
  p {{ margin: 8px 0; }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px auto;
    border: 1px solid #d5d8dc;
    border-radius: 4px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 9.5pt;
  }}
  th {{
    background-color: #1a5276;
    color: white;
    padding: 7px 10px;
    text-align: left;
    border: 1px solid #1a5276;
  }}
  td {{
    padding: 6px 10px;
    border: 1px solid #d5d8dc;
  }}
  tr:nth-child(even) td {{ background-color: #eaf4fb; }}
  code {{
    background: #f4f6f7;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 9pt;
    color: #c0392b;
  }}
  pre {{
    background: #f4f6f7;
    border: 1px solid #d5d8dc;
    border-left: 4px solid #2e86c1;
    padding: 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.4;
  }}
  pre code {{
    background: none;
    padding: 0;
    color: #1a1a1a;
  }}
  blockquote {{
    border-left: 4px solid #f39c12;
    margin: 14px 0;
    padding: 8px 16px;
    background: #fef9e7;
    border-radius: 0 4px 4px 0;
  }}
  hr {{
    border: none;
    border-top: 1px solid #d5d8dc;
    margin: 24px 0;
  }}
  strong {{ color: #17202a; }}
  .page-break {{ page-break-after: always; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    print("Generating PDF with xhtml2pdf...")

    def link_callback(uri, rel):
        """Resolve font and image URIs to local absolute paths."""
        # Handle local font files
        for font_path in [NIRMALA_REGULAR, NIRMALA_BOLD]:
            if os.path.basename(font_path) in uri:
                return str(Path(font_path).resolve())
        return uri

    with open(str(output_path), "wb") as pdf_file:
        result = pisa.CreatePDF(html, dest=pdf_file, encoding='utf-8',
                                link_callback=link_callback)
    
    if result.err:
        print(f"PDF generation had errors: {result.err}")
    else:
        print(f"\nDONE! PDF saved to: {output_path}")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)  # ensure we're in repo root
    reports = [
        (Path("report/final_report.md"), Path("report/Rajeev_Kandpal_Technical_Report.pdf")),
        (Path("report/experiment_details.md"), Path("report/Part1__Experiment_Details.pdf"))
    ]
    for md, pdf in reports:
        md_to_pdf(md, pdf)
