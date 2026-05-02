from __future__ import annotations

import csv
import html
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
PAGES_GALLERY_DIR = PAGES_ASSETS_DIR / "best100"
PAGES_HTML_OUT = PAGES_DIR / "index.html"
PAGES_REPORT_OUT = PAGES_DIR / "report.html"
PAGES_CSS_OUT = PAGES_DIR / "styles.css"
PAGES_JS_OUT = PAGES_DIR / "app.js"
PAGES_PDF_OUT = PAGES_DIR / "final_project_progress_report.pdf"
PAGES_IMAGE_OUT = PAGES_ASSETS_DIR / "best_grid.png"

SOURCE_IMAGE = ROOT / "showcase_images" / "v21_ensemble_comparison" / "best_grid.png"
SHOWCASE_DIR = ROOT / "showcase_images" / "best100_clahe_balanced"
SHOWCASE_METADATA = SHOWCASE_DIR / "metadata.csv"


REPORT_CSS = """
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
  page-break-inside: avoid;
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


PAGE_CSS = """
:root {
  color-scheme: light;
  --bg: #f5f5f7;
  --panel: #ffffff;
  --ink: #1d1d1f;
  --muted: #6e6e73;
  --line: rgba(29, 29, 31, 0.12);
  --accent: #0066cc;
  --accent-ink: #004a99;
  --shadow: 0 22px 70px rgba(0, 0, 0, 0.08);
  --radius: 22px;
}
* {
  box-sizing: border-box;
}
html {
  scroll-behavior: smooth;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}
img {
  max-width: 100%;
  display: block;
}
a {
  color: inherit;
  text-decoration: none;
}
.site-nav {
  position: sticky;
  top: 0;
  z-index: 20;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  background: rgba(245, 245, 247, 0.82);
  backdrop-filter: blur(20px);
}
.nav-inner {
  max-width: 1180px;
  margin: 0 auto;
  min-height: 58px;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}
.brand {
  font-size: 15px;
  font-weight: 700;
  white-space: nowrap;
}
.nav-links {
  display: flex;
  align-items: center;
  gap: 22px;
  font-size: 13px;
  color: var(--muted);
}
.nav-links a:hover {
  color: var(--ink);
}
.hero {
  max-width: 1180px;
  margin: 0 auto;
  padding: 78px 24px 54px;
}
.eyebrow {
  color: var(--accent);
  font-size: 14px;
  font-weight: 700;
  margin: 0 0 16px;
}
h1 {
  max-width: 960px;
  margin: 0;
  font-size: clamp(48px, 7vw, 96px);
  line-height: 0.98;
  letter-spacing: 0;
}
.hero-copy {
  max-width: 760px;
  margin: 24px 0 0;
  color: #424245;
  font-size: clamp(18px, 2vw, 24px);
  line-height: 1.38;
}
.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 30px;
}
.button {
  min-height: 44px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0 20px;
  font-size: 15px;
  font-weight: 650;
  border: 1px solid transparent;
}
.button.primary {
  color: #fff;
  background: var(--accent);
}
.button.secondary {
  color: var(--accent-ink);
  background: rgba(0, 102, 204, 0.08);
  border-color: rgba(0, 102, 204, 0.18);
}
.metric-strip {
  margin-top: 44px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.metric-card,
.artifact-card,
.feature-card,
.gallery-card,
.timeline-item,
.note-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.04);
}
.metric-card {
  padding: 18px;
}
.metric-value {
  font-size: 34px;
  font-weight: 760;
  letter-spacing: 0;
}
.metric-label {
  margin-top: 6px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.35;
}
main {
  overflow: hidden;
}
section {
  max-width: 1180px;
  margin: 0 auto;
  padding: 58px 24px;
}
.section-head {
  max-width: 780px;
  margin-bottom: 26px;
}
.section-kicker {
  margin: 0 0 10px;
  color: var(--accent);
  font-size: 13px;
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
h2 {
  margin: 0;
  font-size: clamp(32px, 4.6vw, 58px);
  line-height: 1.03;
  letter-spacing: 0;
}
.section-lede {
  margin: 16px 0 0;
  color: #424245;
  font-size: 18px;
  line-height: 1.55;
}
.timeline {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.timeline-item {
  padding: 20px;
}
.timeline-label {
  color: var(--accent);
  font-size: 13px;
  font-weight: 750;
}
.timeline-item h3,
.artifact-card h3,
.feature-card h3,
.gallery-card h3 {
  margin: 8px 0 0;
  font-size: 18px;
  line-height: 1.25;
}
.timeline-item p,
.artifact-card p,
.feature-card p,
.gallery-card p,
.note-panel p {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.48;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}
.feature-card {
  overflow: hidden;
}
.feature-card img {
  width: 100%;
  aspect-ratio: 1228 / 598;
  object-fit: cover;
  border-bottom: 1px solid var(--line);
  cursor: zoom-in;
}
.feature-card-body {
  padding: 18px;
}
.score-line {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  color: #424245;
  font-size: 12px;
  font-weight: 650;
  padding: 6px 9px;
  background: #fafafa;
}
.gallery-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin: 4px 0 18px;
}
.gallery-count {
  color: var(--muted);
  font-size: 14px;
}
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.gallery-card {
  overflow: hidden;
}
.gallery-card[hidden] {
  display: none;
}
.gallery-card img {
  width: 100%;
  aspect-ratio: 1228 / 598;
  object-fit: cover;
  border-bottom: 1px solid var(--line);
  cursor: zoom-in;
}
.gallery-card-body {
  padding: 14px;
}
.gallery-card h3 {
  font-size: 15px;
  overflow-wrap: anywhere;
}
.gallery-card p {
  margin: 8px 0 0;
  font-size: 12px;
}
.artifact-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.artifact-card {
  padding: 20px;
}
.artifact-card a {
  color: var(--accent-ink);
  font-weight: 700;
}
.note-panel {
  padding: 24px;
}
.footer {
  max-width: 1180px;
  margin: 0 auto;
  padding: 34px 24px 46px;
  color: var(--muted);
  font-size: 13px;
  border-top: 1px solid var(--line);
}
.lightbox {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 28px;
  background: rgba(245, 245, 247, 0.94);
}
.lightbox.open {
  display: flex;
}
.lightbox img {
  max-height: 88vh;
  width: auto;
  border-radius: 18px;
  box-shadow: var(--shadow);
}
.lightbox button {
  position: absolute;
  top: 18px;
  right: 18px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 999px;
  min-width: 42px;
  min-height: 42px;
  font-size: 22px;
  cursor: pointer;
}
@media (max-width: 900px) {
  .nav-links {
    display: none;
  }
  .metric-strip,
  .timeline,
  .feature-grid,
  .gallery-grid,
  .artifact-grid {
    grid-template-columns: 1fr;
  }
  .hero {
    padding-top: 56px;
  }
  section {
    padding-top: 42px;
    padding-bottom: 42px;
  }
}
"""


PAGE_JS = """
document.addEventListener("DOMContentLoaded", () => {
  const galleryCards = Array.from(document.querySelectorAll(".gallery-card"));
  const toggle = document.querySelector("[data-toggle-gallery]");
  const galleryCount = document.querySelector("[data-gallery-count]");
  let expanded = false;

  function updateGallery() {
    galleryCards.forEach((card, index) => {
      card.hidden = !expanded && index >= 12;
    });
    if (toggle) {
      toggle.textContent = expanded ? "Show first 12" : "Show all 100 panels";
      toggle.setAttribute("aria-expanded", String(expanded));
    }
    if (galleryCount) {
      galleryCount.textContent = expanded ? "Showing all 100 panels" : "Showing 12 of 100 panels";
    }
  }

  if (toggle) {
    toggle.addEventListener("click", () => {
      expanded = !expanded;
      updateGallery();
    });
  }
  updateGallery();

  const lightbox = document.querySelector(".lightbox");
  const lightboxImage = document.querySelector(".lightbox img");
  const lightboxClose = document.querySelector(".lightbox button");

  function openLightbox(src, alt) {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = src;
    lightboxImage.alt = alt || "Virtual staining comparison panel";
    lightbox.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    if (!lightbox || !lightboxImage) return;
    lightbox.classList.remove("open");
    lightboxImage.src = "";
    document.body.style.overflow = "";
  }

  document.querySelectorAll("[data-lightbox]").forEach((image) => {
    image.addEventListener("click", () => openLightbox(image.src, image.alt));
  });
  if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
  if (lightbox) {
    lightbox.addEventListener("click", (event) => {
      if (event.target === lightbox) closeLightbox();
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
  });
});
"""


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def f4(value: str) -> str:
    return f"{float(value):.4f}"


def f3(value: str) -> str:
    return f"{float(value):.3f}"


def load_showcase_rows() -> list[dict[str, str]]:
    with SHOWCASE_METADATA.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["filename"] = Path(row["panel"]).name
    return rows


def render_report_html(
    body: str,
    include_tools: bool = False,
    pages_paths: bool = False,
    report_relative_paths: bool = False,
) -> str:
    if pages_paths:
        body = body.replace(
            'src="showcase_images/v21_ensemble_comparison/best_grid.png"',
            'src="assets/best_grid.png"',
        )
        body = body.replace(
            'src="showcase_images/best100_clahe_balanced/featured/',
            'src="assets/best100/',
        )
    if report_relative_paths:
        body = body.replace('src="showcase_images/', 'src="../showcase_images/')
    tools = ""
    if include_tools:
        tools = (
            '<div class="site-tools">'
            '<strong>Full project report.</strong> '
            '<a href="index.html">Open the visual research page</a> or '
            '<a href="final_project_progress_report.pdf">download the PDF version</a>.'
            "</div>\n"
        )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Virtual H&E Staining Capstone Progress Report</title>
<style>{REPORT_CSS}</style>
</head>
<body>
{tools}{body}
</body>
</html>
"""


def metric_stats(rows: list[dict[str, str]]) -> dict[str, str]:
    slides = {row["prefix"].split("_patch_")[0] for row in rows}
    mean_quality = sum(float(row["content_quality_score"]) for row in rows) / len(rows)
    mean_ssim = sum(float(row["ssim_full_rgb"]) for row in rows) / len(rows)
    mean_content = sum(float(row["content_fraction"]) for row in rows) / len(rows)
    mean_delta = sum(float(row["delta_vs_old"]) for row in rows) / len(rows)
    return {
        "slide_count": str(len(slides)),
        "mean_quality": f"{mean_quality:.3f}",
        "mean_ssim": f"{mean_ssim:.3f}",
        "mean_content": f"{mean_content:.3f}",
        "mean_delta": f"{mean_delta:+.3f}",
    }


def feature_card(row: dict[str, str]) -> str:
    rank = int(row["rank"])
    src = f"assets/best100/{e(row['filename'])}"
    title = f"#{rank:03d} {row['prefix']}"
    return f"""
    <article class="feature-card">
      <img src="{src}" alt="{e(title)}" loading="lazy" data-lightbox>
      <div class="feature-card-body">
        <h3>{e(title)}</h3>
        <div class="score-line">
          <span class="pill">Quality {f4(row['content_quality_score'])}</span>
          <span class="pill">CLAHE SSIM {f4(row['ssim_full_rgb'])}</span>
          <span class="pill">Tissue {f3(row['content_fraction'])}</span>
        </div>
      </div>
    </article>
"""


def gallery_card(row: dict[str, str]) -> str:
    rank = int(row["rank"])
    src = f"assets/best100/{e(row['filename'])}"
    title = f"#{rank:03d} {row['prefix']}"
    return f"""
    <article class="gallery-card">
      <img src="{src}" alt="{e(title)}" loading="lazy" data-lightbox>
      <div class="gallery-card-body">
        <h3>{e(title)}</h3>
        <p>Quality {f4(row['content_quality_score'])} - CLAHE SSIM {f4(row['ssim_full_rgb'])} - tissue {f3(row['content_fraction'])}</p>
      </div>
    </article>
"""


def build_pages_home(rows: list[dict[str, str]]) -> str:
    stats = metric_stats(rows)
    featured = [row for row in rows if row.get("featured_panel")][:6]
    feature_cards = "\n".join(feature_card(row) for row in featured).strip()
    gallery_cards = "\n".join(gallery_card(row) for row in rows).strip()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Virtual H&E Staining Capstone</title>
<meta name="description" content="Final IIT Hyderabad Biomedical Engineering capstone report for virtual H&E staining from unstained tissue patches.">
<link rel="stylesheet" href="styles.css">
</head>
<body>
<nav class="site-nav" aria-label="Primary navigation">
  <div class="nav-inner">
    <a class="brand" href="#top">Virtual H&E</a>
    <div class="nav-links">
      <a href="#results">Results</a>
      <a href="#method">Method</a>
      <a href="#showcase">Showcase</a>
      <a href="#artifacts">Artifacts</a>
      <a href="final_project_progress_report.pdf">PDF</a>
    </div>
  </div>
</nav>

<header class="hero" id="top">
  <p class="eyebrow">IIT Hyderabad Biomedical Engineering Capstone</p>
  <h1>Virtual H&E Staining from Unstained Tissue Images</h1>
  <p class="hero-copy">A complete research pipeline for aligning unstained and stained histology patches, training Pix2Pix-style virtual staining models, and shipping a final v20/v22A/v21B ensemble app with a reproducible report.</p>
  <div class="hero-actions">
    <a class="button primary" href="final_project_progress_report.pdf">Download report</a>
    <a class="button secondary" href="report.html">Read full HTML report</a>
    <a class="button secondary" href="#showcase">View best-100 gallery</a>
  </div>
  <div class="metric-strip" id="results">
    <div class="metric-card">
      <div class="metric-value">0.7838</div>
      <div class="metric-label">Final CLAHE validation SSIM</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">25.16</div>
      <div class="metric-label">Best final PSNR in dB</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">0.8794</div>
      <div class="metric-label">Best final Pearson correlation</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">100</div>
      <div class="metric-label">Best newer-registration panels in this gallery</div>
    </div>
  </div>
</header>

<main>
  <section id="method">
    <div class="section-head">
      <p class="section-kicker">Pipeline</p>
      <h2>Registration first, model selection second.</h2>
      <p class="section-lede">The major project lesson was that virtual staining quality is limited by alignment quality. The final path uses CLAHE-preprocessed TV-L1 optical flow, content-aware pair selection, strong single-model baselines, and a conservative weighted ensemble.</p>
    </div>
    <div class="timeline">
      <article class="timeline-item">
        <div class="timeline-label">01</div>
        <h3>Pair registration</h3>
        <p>CLAHE TV-L1 improved the all-pair registration baseline from 0.4134 to 0.5045 mean SSIM.</p>
      </article>
      <article class="timeline-item">
        <div class="timeline-label">02</div>
        <h3>Filtered training set</h3>
        <p>The top-1000 content-quality set favored tissue-rich patches with positive registration gain.</p>
      </article>
      <article class="timeline-item">
        <div class="timeline-label">03</div>
        <h3>Final candidates</h3>
        <p>v20 remained strong on old registration; v22A and v21B were strongest on the newer CLAHE distribution.</p>
      </article>
      <article class="timeline-item">
        <div class="timeline-label">04</div>
        <h3>Balanced ensemble</h3>
        <p>The default app mode is 0.20*v20 + 0.60*v22A + 0.20*v21B with TTA4.</p>
      </article>
    </div>
  </section>

  <section id="showcase">
    <div class="section-head">
      <p class="section-kicker">Qualitative Evidence</p>
      <h2>Six featured examples from the best-100 newer registration set.</h2>
      <p class="section-lede">Each panel contains the registered unstained input, the virtual H&E prediction, and the real stained target. The full gallery below includes all 100 generated panels.</p>
    </div>
    <div class="feature-grid">
      {feature_cards}
    </div>
  </section>

  <section id="gallery">
    <div class="section-head">
      <p class="section-kicker">Best-100 Gallery</p>
      <h2>Traceable panels with registration metadata.</h2>
      <p class="section-lede">These 100 examples span {e(stats['slide_count'])} slides. Their mean content-quality score is {e(stats['mean_quality'])}, mean CLAHE registration SSIM is {e(stats['mean_ssim'])}, mean tissue content is {e(stats['mean_content'])}, and mean SSIM delta over old registration is {e(stats['mean_delta'])}.</p>
    </div>
    <div class="gallery-toolbar">
      <span class="gallery-count" data-gallery-count>Showing 12 of 100 panels</span>
      <button class="button secondary" type="button" data-toggle-gallery aria-expanded="false">Show all 100 panels</button>
    </div>
    <div class="gallery-grid" data-gallery>
      {gallery_cards}
    </div>
  </section>

  <section id="artifacts">
    <div class="section-head">
      <p class="section-kicker">Repository</p>
      <h2>Everything needed to inspect or reproduce the final result.</h2>
      <p class="section-lede">The page keeps the high-level story readable, while the PDF and repository files preserve the exact logs, scripts, model manifest, and generated images.</p>
    </div>
    <div class="artifact-grid">
      <article class="artifact-card">
        <h3><a href="final_project_progress_report.pdf">Detailed PDF report</a></h3>
        <p>Human-readable project history, results, decisions, challenges, citations, and qualitative examples.</p>
      </article>
      <article class="artifact-card">
        <h3><a href="report.html">Full HTML report</a></h3>
        <p>The same report in browser form, useful for quick reading and linking sections.</p>
      </article>
      <article class="artifact-card">
        <h3><a href="assets/best100/001_AS-5198-23-Z36_patch_38912_49152.jpg">Best panel asset</a></h3>
        <p>Direct access to the generated visual comparison assets used by the page and PDF.</p>
      </article>
    </div>
  </section>

  <section>
    <div class="note-panel">
      <h2>Final handoff position</h2>
      <p>The current best reproducible baseline is the CLAHE TV-L1 plus balanced v20/v22A/v21B TTA4 ensemble. The next serious research step is not another small architecture tweak; it is broader validation with stronger registration QA, more slide diversity, and clinical review of failure modes.</p>
    </div>
  </section>
</main>

<footer class="footer">
  <p>Nishant and Arin. Professor: Renu John. TA: Sarfaraz. IIT Hyderabad, Biomedical Engineering.</p>
</footer>

<div class="lightbox" aria-modal="true" role="dialog">
  <button type="button" aria-label="Close image preview">x</button>
  <img src="" alt="">
</div>
<script src="app.js"></script>
</body>
</html>
"""


def copy_showcase_assets(rows: list[dict[str, str]]) -> None:
    PAGES_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    for stale in PAGES_GALLERY_DIR.glob("*.jpg"):
        stale.unlink()
    for row in rows:
        src = ROOT / row["panel"]
        shutil.copy2(src, PAGES_GALLERY_DIR / src.name)


def main() -> None:
    rows = load_showcase_rows()
    text = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )

    pdf_report = render_report_html(body, include_tools=False, pages_paths=False)
    html_report = render_report_html(body, include_tools=False, report_relative_paths=True)
    HTML_OUT.write_text(html_report, encoding="utf-8")
    HTML(string=pdf_report, base_url=str(ROOT)).write_pdf(PDF_OUT)

    PAGES_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    copy_showcase_assets(rows)
    shutil.copy2(PDF_OUT, PAGES_PDF_OUT)
    shutil.copy2(SOURCE_IMAGE, PAGES_IMAGE_OUT)

    PAGES_HTML_OUT.write_text(build_pages_home(rows), encoding="utf-8")
    PAGES_REPORT_OUT.write_text(render_report_html(body, include_tools=True, pages_paths=True), encoding="utf-8")
    PAGES_CSS_OUT.write_text(PAGE_CSS, encoding="utf-8")
    PAGES_JS_OUT.write_text(PAGE_JS, encoding="utf-8")
    (PAGES_DIR / ".nojekyll").write_text("", encoding="utf-8")

    print(PDF_OUT)
    print(PAGES_HTML_OUT)
    print(PAGES_REPORT_OUT)
    print(PAGES_GALLERY_DIR)


if __name__ == "__main__":
    main()
