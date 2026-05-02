from __future__ import annotations

import shutil
from pathlib import Path

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
SOURCE = REPORT_DIR / "final_project_progress_report.md"
HTML_OUT = REPORT_DIR / "final_project_progress_report.html"
PDF_OUT = REPORT_DIR / "final_project_progress_report.pdf"
PAGES_DIR = ROOT / "docs"
PAGES_ASSETS_DIR = PAGES_DIR / "assets"
PAGES_HTML_OUT = PAGES_DIR / "index.html"
PAGES_PDF_OUT = PAGES_DIR / "final_project_progress_report.pdf"
PAGES_IMAGE_OUT = PAGES_ASSETS_DIR / "best_grid.png"
SOURCE_IMAGE = ROOT / "showcase_images" / "v21_ensemble_comparison" / "best_grid.png"


CSS = """
@page {
  size: A4;
  margin: 18mm 15mm 18mm 15mm;
  @bottom-center {
    content: "Virtual H&E Staining Capstone Progress Report - Page " counter(page);
    font-size: 8.5pt;
    color: #666;
  }
}
body {
  font-family: Arial, Helvetica, sans-serif;
  color: #1f2933;
  font-size: 10.5pt;
  line-height: 1.48;
}
h1 {
  font-size: 26pt;
  color: #12355b;
  margin: 0 0 14pt 0;
  padding-bottom: 10pt;
  border-bottom: 3px solid #2f80ed;
}
h2 {
  font-size: 17pt;
  color: #12355b;
  margin-top: 22pt;
  padding-top: 4pt;
  border-top: 1px solid #d8e2ef;
}
h3 {
  font-size: 12.5pt;
  color: #264b73;
  margin-top: 14pt;
}
p {
  margin: 7pt 0;
}
blockquote {
  border-left: 4px solid #2f80ed;
  margin: 10pt 0;
  padding: 6pt 10pt;
  background: #f3f8ff;
}
code {
  font-family: "Courier New", monospace;
  background: #f4f4f4;
  padding: 1pt 3pt;
  border-radius: 3px;
}
pre {
  background: #f4f4f4;
  border: 1px solid #ddd;
  padding: 8pt;
  white-space: pre-wrap;
  font-size: 9pt;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 9pt 0 13pt 0;
  font-size: 9.2pt;
}
th {
  background: #eaf2ff;
  color: #12355b;
  font-weight: 700;
}
th, td {
  border: 1px solid #c9d6e2;
  padding: 5pt 6pt;
  vertical-align: top;
}
tr:nth-child(even) td {
  background: #fbfdff;
}
hr {
  border: none;
  border-top: 1px solid #d8e2ef;
  margin: 16pt 0;
}
.wide-img {
  width: 100%;
  height: auto;
  border: 1px solid #c9d6e2;
  margin-top: 8pt;
}
a {
  color: #1b66b1;
  text-decoration: none;
}
.site-tools {
  display: none;
}
@media screen {
  body {
    max-width: 980px;
    margin: 0 auto;
    padding: 28px 22px 60px;
    background: #ffffff;
  }
  .site-tools {
    display: block;
    background: #f3f8ff;
    border: 1px solid #c9d6e2;
    padding: 10px 12px;
    margin-bottom: 18px;
    font-size: 10pt;
  }
}
"""


def render_html(body: str, image_path: str, include_tools: bool) -> str:
    body = body.replace(
        'src="showcase_images/v21_ensemble_comparison/best_grid.png"',
        f'src="{image_path}"',
    )
    tools = ""
    if include_tools:
        tools = (
            '<div class="site-tools">'
            '<strong>Project report page.</strong> '
            '<a href="final_project_progress_report.pdf">Download the PDF version</a>.'
            "</div>\n"
        )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Virtual H&E Staining Capstone Progress Report</title>
<style>{CSS}</style>
</head>
<body>
{tools}{body}
</body>
</html>
"""


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    html = render_html(body, "showcase_images/v21_ensemble_comparison/best_grid.png", include_tools=False)
    HTML_OUT.write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(ROOT)).write_pdf(PDF_OUT)
    PAGES_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    page_html = render_html(body, "assets/best_grid.png", include_tools=True)
    PAGES_HTML_OUT.write_text(page_html, encoding="utf-8")
    shutil.copy2(PDF_OUT, PAGES_PDF_OUT)
    shutil.copy2(SOURCE_IMAGE, PAGES_IMAGE_OUT)
    (PAGES_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(PDF_OUT)
    print(PAGES_HTML_OUT)


if __name__ == "__main__":
    main()
