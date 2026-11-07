# Thermodynamic Computer Map

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
