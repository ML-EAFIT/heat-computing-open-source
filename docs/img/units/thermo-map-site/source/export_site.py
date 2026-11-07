#!/usr/bin/env python3
"""Exports the thermodynamic computer map as a static website.

usage: python3 export_site.py <output-folder>

Writes index.html (zoomable map), reference.html (full reference page) and assets/.
Everything is static: no build step, no server code, no data fetches, so the folder
works from any static host and also straight from disk (file://).
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_map as bm      # noqa: E402
import build_zoom as bz     # noqa: E402

DESC = ("Zoomable wiring map of a p-bit thermodynamic sampler on a Tang Nano 9K FPGA: "
        "from the host PC down to block RAM cells and single transistors.")

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="12" fill="#141c25"/>
<g stroke="#7eabff" stroke-width="3" stroke-linecap="round">
<path d="M22 8v8M32 8v8M42 8v8M22 48v8M32 48v8M42 48v8M8 22h8M8 32h8M8 42h8M48 22h8M48 32h8M48 42h8"/>
</g>
<rect x="16" y="16" width="32" height="32" rx="4" fill="#1b5fc9" stroke="#7eabff" stroke-width="2"/>
<circle cx="32" cy="32" r="6" fill="#f0a640"/>
</svg>
"""

HEAD_EXTRA = f"""<meta name="description" content="{bm.E(DESC)}">
<meta name="theme-color" content="#141c25">
<meta property="og:type" content="website">
<meta property="og:title" content="{{title}}">
<meta property="og:description" content="{bm.E(DESC)}">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
"""

README = """# Thermodynamic Computer Map

A static website that maps the p-bit thermodynamic sampler (19 Verilog modules built
for the Sipeed Tang Nano 9K) at every level of abstraction:

* `index.html`: the zoomable map. Scroll or pinch to zoom, drag to pan, click any
  outlined block to fly into it, Esc to go up. It nests 40 schematics, from the host PC
  and board down through the bitstream top, the engines, the p-bit, carry chains, LUTs,
  block RAM and SRAM cells to a single NMOS/PMOS transistor.
* `reference.html`: the same material as one long reference page (UART command table,
  number formats, cycle budgets, synthesis and place-and-route results, signal dictionary).

No build step, no server code, no data fetches. Open `index.html` directly from disk
or serve the folder from any static host.

## View locally

    python3 -m http.server 8000      # then open http://localhost:8000

(Opening `index.html` by double-click also works.)

## Deploy

* **GitHub Pages**: put the folder contents in a repo (root or `docs/`), then
  Settings → Pages → Deploy from a branch → pick the branch and folder. The `.nojekyll`
  file is already included.
* **Netlify / Cloudflare Pages / Vercel**: drag the folder onto the dashboard, or point a
  project at it with no build command and the folder as the publish directory.
* **Any web server**: copy the files as they are.

## Files

    index.html          zoomable map (loads assets/map.css, assets/views.js, assets/map.js)
    reference.html      full reference page (self-contained)
    assets/map.css      map styles, light and dark themes
    assets/views.js     the 40 schematics (SVG) with their zoom links and panel text
    assets/map.js       the zoom engine: camera, re-rooting, level of detail, input
    assets/favicon.svg  icon
    source/             the Python generators that produced all of the above

Fonts come from Google Fonts (IBM Plex); without network access the pages fall back to
system fonts.

## Regenerate

    python3 source/export_site.py <output-folder>

Needs Python 3.8+ only. The synthesis and place-and-route figures are stored in
`source/build_map.py`; they came from Yosys 0.70 (synth_gowin) and nextpnr-himbaechel for
GW1NR-LV9QN88PC6/I5 at 27 MHz.

## Notes

* Transistor-level drawings are textbook structures; Gowin does not publish its cell
  layouts.
* The Gaussian ROM curve is a standard inverse CDF, not the exported table.
* Board pin numbers are the standard Tang Nano 9K ones.
"""


def add_head(html, title):
    return html.replace("<title>", HEAD_EXTRA.replace("{title}", bm.E(title)) + "<title>", 1)


def main(out):
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)

    # ---------- zoomable map, split into html + css + js
    html = bz.build_html()
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    data_m = re.search(r"<script>window\.__TYPES__ = (.*?);</script>", html, re.S)
    js_m = re.search(r"<script>(\s*\(function\(\)\{.*?)</script>", html, re.S)
    html = html.replace(f"<style>{css}</style>", '<link rel="stylesheet" href="assets/map.css">')
    html = html.replace(data_m.group(0), '<script src="assets/views.js"></script>')
    html = html.replace(js_m.group(0), '<script src="assets/map.js"></script>')
    html = add_head(html, "Thermodynamic Computer Map")
    files = {
        "index.html": html,
        "assets/map.css": css.strip() + "\n",
        "assets/views.js": "window.__TYPES__ = " + data_m.group(1) + ";\n",
        "assets/map.js": js_m.group(1).strip() + "\n",
        "assets/favicon.svg": FAVICON,
    }

    # ---------- reference page, with links back to the map
    ref = bm.build()
    ref = ref.replace("<title>Thermodynamic Computer Wiring Map</title>", "<title>Thermodynamic Computer Reference</title>")
    ref = add_head(ref, "Thermodynamic Computer Reference")
    ref = ref.replace('<nav class="idx" aria-label="Sections"><h2>Map index</h2><ol>',
                      '<nav class="idx" aria-label="Sections"><h2>Map index</h2><p class="back"><a href="index.html">← Zoomable map</a></p><ol>', 1)
    ref = ref.replace('<p class="eyebrow">Tang Nano 9K',
                      '<p class="eyebrow"><a href="index.html">Open the zoomable map</a> · Tang Nano 9K', 1)
    ref = ref.replace('<h1>Thermodynamic Computer Wiring Map</h1>', '<h1>Thermodynamic Computer Reference</h1>', 1)
    ref = ref.replace("</style>", "nav.idx .back{margin:0 0 12px 6px;font-size:.88rem}\nnav.idx .back a{display:inline;padding:0;text-decoration:none;font-weight:600;color:var(--signal)}\n</style>", 1)
    files["reference.html"] = ref

    files["README.md"] = README
    files[".nojekyll"] = ""

    for name, content in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            fh.write(content)

    # ---------- generators, so the site can be rebuilt
    src = os.path.join(out, "source")
    if os.path.abspath(src) != HERE:
        os.makedirs(src, exist_ok=True)
        for name in ("build_map.py", "zoom_views.py", "build_zoom.py", "export_site.py"):
            shutil.copy(os.path.join(HERE, name), os.path.join(src, name))

    for name in sorted(files):
        print(f"{name:22s} {len(files[name].encode('utf-8')):>9,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
