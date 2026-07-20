#!/usr/bin/env python3
"""Build docs/DASHBOARD.pdf from docs/DASHBOARD.md — pandoc (md→standalone HTML) then
headless Chrome (HTML→PDF). No LaTeX / no extra installs. A PDF has no preview size cap
and renders embedded images natively, so it displays where the big markdown won't.

    python build_dashboard.py   # first (refresh the md + embedded charts)
    python build_pdf.py         # then (md -> pdf)
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tempfile

DOCS = "docs"
MD = os.path.join(DOCS, "DASHBOARD.md")
PDF = os.path.join(DOCS, "DASHBOARD.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main() -> int:
    if not os.path.exists(MD):
        print(f"missing {MD} — run build_dashboard.py first."); return 1
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("pandoc not found."); return 1
    if not os.path.exists(CHROME):
        print(f"Chrome not found at {CHROME}."); return 1

    html = os.path.join(tempfile.gettempdir(), "dashboard.html")
    subprocess.run([pandoc, MD, "-s", "-o", html,
                    "--metadata", "title=AI × Modern Mercantilism — Dashboard"], check=True)
    print(f"pandoc → {html} ({os.path.getsize(html)/1e6:.1f} MB)")

    profile = tempfile.mkdtemp(prefix="chromepdf_")
    abspdf = os.path.abspath(PDF)
    cmd = [CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--no-first-run",
           "--no-default-browser-check", "--virtual-time-budget=20000",
           "--run-all-compositor-stages-before-draw", "--no-pdf-header-footer",
           f"--user-data-dir={profile}", f"--print-to-pdf={abspdf}",
           f"file://{html}"]
    try:
        # virtual-time-budget makes Chrome finish+exit; the timeout is just a safety net
        # (if it writes the PDF then lingers, we kill it — the file is already on disk).
        subprocess.run(cmd, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print("  (Chrome lingered past the timeout and was killed — PDF should already be written)")

    if os.path.exists(PDF) and os.path.getsize(PDF) > 1000:
        raw = open(PDF, "rb").read()
        ok = raw[:5] == b"%PDF-" and b"%%EOF" in raw[-2048:]
        print(f"✓ wrote {PDF} ({len(raw)/1e6:.2f} MB) — {'valid' if ok else 'WARNING: may be truncated'}")
        return 0 if ok else 2
    print("✗ no PDF produced"); return 1


if __name__ == "__main__":
    sys.exit(main())
