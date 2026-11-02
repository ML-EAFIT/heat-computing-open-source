#!/usr/bin/env python3
"""Grep PDF text with page numbers: cite.py <txtfile> <regex> [context].
Text files come from `pdftotext -layout` (pages separated by form feeds)."""
import re, sys
path, pat = sys.argv[1], re.compile(sys.argv[2], re.I)
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
pages = open(path, encoding="utf-8", errors="replace").read().split("\f")
for pno, page in enumerate(pages, 1):
    lines = page.splitlines()
    for i, ln in enumerate(lines):
        if pat.search(ln):
            lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
            print(f"p.{pno}: " + " | ".join(l.strip() for l in lines[lo:hi]))
