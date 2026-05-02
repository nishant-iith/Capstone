from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
SOURCE = REPORT_DIR / "final_project_progress_report.md"
HTML_OUT = REPORT_DIR / "final_project_progress_report.html"
PDF_OUT = REPORT_DIR / "final_project_progress_report.pdf"


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
"""


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Virtual H&E Staining Capstone Progress Report</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(ROOT)).write_pdf(PDF_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
