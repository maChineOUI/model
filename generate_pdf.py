#!/usr/bin/env /opt/homebrew/bin/python3.12
"""
generate_pdf.py
Convert report.md → report.pdf.
Math is rendered via matplotlib (SVG, base64-embedded).
pmatrix environments fall back to latex2mathml MathML.
"""

import re
import io
import base64
import pathlib
import html as html_mod
import markdown as md_lib
import latex2mathml.converter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent

# ── 1.  LaTeX → matplotlib-compatible LaTeX ────────────────────────────────────

# Commands that need substitution
_FIXES = [
    (re.compile(r'\\le\b'),               r'\\leq'),
    (re.compile(r'\\ge\b'),               r'\\geq'),
    (re.compile(r'\\tfrac\b'),            r'\\frac'),
    (re.compile(r'\\implies\b'),          r'\\Rightarrow'),
    (re.compile(r'\\bigl\b'),             r'\\left'),
    (re.compile(r'\\bigr\b'),             r'\\right'),
    (re.compile(r'\\Bigl\b'),             r'\\left'),
    (re.compile(r'\\Bigr\b'),             r'\\right'),
    (re.compile(r'\\[Bb]ig([lrmm]?)\b'), r''),
    # Math-mode spacing commands unsupported by mathtext
    (re.compile(r'\\;'),                  r'\\,'),
]

_RE_PMATRIX = re.compile(r'\\begin\{pmatrix\}', re.DOTALL)
_RE_BOXED   = re.compile(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')


def _fix_latex(tex: str) -> tuple[str, bool]:
    """
    Return (fixed_tex, is_boxed).
    Transforms unsupported commands into matplotlib-mathtext equivalents.
    """
    is_boxed = bool(_RE_BOXED.search(tex))
    # Strip \boxed{} wrapper(s)
    while _RE_BOXED.search(tex):
        tex = _RE_BOXED.sub(r'\1', tex)
    for pat, repl in _FIXES:
        tex = pat.sub(repl, tex)
    return tex.strip(), is_boxed


# ── 2.  Rendering ──────────────────────────────────────────────────────────────

_SVG_CACHE: dict[tuple, str] = {}

# Strip SVG preamble, keep only <svg>...</svg>
_RE_SVG_STRIP = re.compile(r'(?s)^.*?(<svg\b)', re.DOTALL)


def _render_svg(tex: str, display: bool) -> str:
    """Render tex to an SVG string via matplotlib (cached)."""
    key = (tex, display)
    if key in _SVG_CACHE:
        return _SVG_CACHE[key]

    fontsize = 12 if display else 10.5
    pad      = 0.05 if display else 0.03

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0.0, 0.0, f'${tex}$', fontsize=fontsize,
             usetex=False, ha='left', va='baseline')
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight',
                transparent=True, pad_inches=pad)
    plt.close(fig)

    svg_raw = buf.getvalue().decode('utf-8')
    # Keep only the <svg> element
    svg = _RE_SVG_STRIP.sub(r'\1', svg_raw)
    _SVG_CACHE[key] = svg
    return svg


def _svg_to_uri(svg: str) -> str:
    b64 = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    return f'data:image/svg+xml;base64,{b64}'


def _render_pmatrix_block(tex: str) -> str:
    """Convert a formula containing \\begin{pmatrix} to an HTML flex block."""
    m = re.search(r'\\begin\{pmatrix\}([\s\S]*?)\\end\{pmatrix\}', tex)
    if not m:
        return f'<code class="math-err-d">{html_mod.escape(tex)}</code>'

    before = tex[:m.start()].strip()
    after  = tex[m.end():].strip()
    inner  = m.group(1).strip()

    # ── Render formula before the matrix (e.g. "A_{\text{FTCS}} =")
    before_html = ''
    if before:
        fixed_b, _ = _fix_latex(before)
        try:
            svg = _render_svg(fixed_b, display=False)
            before_html = (
                f'<img class="math-img-inline" src="{_svg_to_uri(svg)}" alt="">'
                '<span class="mx-eq">&nbsp;</span>'
            )
        except Exception:
            before_html = html_mod.escape(before) + '&nbsp;'

    # ── Parse rows and render each cell
    rows_raw = [r.strip() for r in re.split(r'\\\\', inner) if r.strip()]
    trs = ''
    for row_raw in rows_raw:
        cells = [c.strip() for c in row_raw.split('&')]
        tds = ''
        for cell in cells:
            if not cell:
                tds += '<td class="mx-cell"></td>'
            else:
                fixed_c, _ = _fix_latex(cell)
                try:
                    svg = _render_svg(fixed_c, display=False)
                    tds += (
                        '<td class="mx-cell">'
                        f'<img class="math-img-inline" src="{_svg_to_uri(svg)}" alt="">'
                        '</td>'
                    )
                except Exception:
                    tds += f'<td class="mx-cell">{html_mod.escape(cell)}</td>'
        trs += f'<tr>{tds}</tr>'

    # ── After punctuation (e.g. ".")
    after_html = html_mod.escape(after) if after else ''

    return (
        '<div class="mx-container">'
        f'{before_html}'
        '<span class="mx-wrap">'
        '<span class="mx-lbr"></span>'
        f'<table class="mx-tbl">{trs}</table>'
        '<span class="mx-rbr"></span>'
        '</span>'
        f'{after_html}'
        '</div>'
    )


def _math_to_img(tex: str, display: bool) -> tuple[str, bool]:
    """
    Return (html_fragment, is_boxed).
    Uses HTML table rendering for pmatrix; matplotlib SVG for everything else.
    """
    # pmatrix: HTML table with CSS brackets
    if _RE_PMATRIX.search(tex):
        return _render_pmatrix_block(tex), False

    fixed, is_boxed = _fix_latex(tex)
    try:
        svg = _render_svg(fixed, display)
    except Exception:
        # Ultimate fallback: pre-format as code
        e = html_mod.escape(tex)
        cls = 'math-err-d' if display else 'math-err-i'
        return f'<code class="{cls}">{e}</code>', is_boxed

    uri = _svg_to_uri(svg)
    if display:
        cls = 'math-img-display'
        return f'<img class="{cls}" src="{uri}" alt="">', is_boxed
    else:
        cls = 'math-img-inline'
        return f'<img class="{cls}" src="{uri}" alt="">', False


# ── 3.  Document pre-processing ────────────────────────────────────────────────

_RE_FENCE   = re.compile(r'(```[\s\S]*?```|`[^`\n]+`)', re.DOTALL)
_RE_DISPLAY = re.compile(r'\$\$([\s\S]*?)\$\$',          re.DOTALL)
_RE_INLINE  = re.compile(r'(?<!\$)\$([^$\n]+?)\$(?!\$)')

_RE_TAG     = re.compile(r'\\tag\{([^}]+)\}')


def _extract_tag(tex: str) -> tuple[str, str | None]:
    m = _RE_TAG.search(tex)
    if m:
        return _RE_TAG.sub('', tex).strip(), m.group(1)
    return tex.strip(), None


def _render_display_block(m: re.Match) -> str:
    # Collapse internal newlines / extra whitespace to single space
    raw = re.sub(r'\s+', ' ', m.group(1)).strip()
    tex, tag = _extract_tag(raw)
    img, is_boxed = _math_to_img(tex, display=True)
    extra_cls = ' eq-boxed' if is_boxed else ''
    if tag:
        return (
            f'\n\n<div class="eq-row{extra_cls}">'
            f'<div class="eq-math">{img}</div>'
            f'<div class="eq-num">({tag})</div>'
            f'</div>\n\n'
        )
    return f'\n\n<div class="eq-block{extra_cls}">{img}</div>\n\n'


def _render_inline_math(m: re.Match) -> str:
    tex = m.group(1).strip()
    if not tex or re.fullmatch(r'[\\;,\s]+', tex):
        return ''  # pure spacing — skip
    img, _ = _math_to_img(tex, display=False)
    return img


def preprocess(src: str) -> str:
    fence_map: dict[str, str] = {}
    counter = [0]

    def protect(m):
        key = f'\x00F{counter[0]}\x00'
        fence_map[key] = m.group(1)
        counter[0] += 1
        return key

    src = _RE_FENCE.sub(protect, src)
    src = _RE_DISPLAY.sub(_render_display_block, src)
    src = _RE_INLINE.sub(_render_inline_math, src)
    for key, val in fence_map.items():
        src = src.replace(key, val)
    return src


# ── 4.  Markdown → HTML ────────────────────────────────────────────────────────

def to_html(src: str) -> str:
    return md_lib.markdown(
        src,
        extensions=["tables", "fenced_code", "attr_list"],
    )


# ── 5.  HTML post-processing ───────────────────────────────────────────────────

def _wrap_title_block(html: str) -> str:
    pat = re.compile(
        r'(<h1>.*?</h1>)'
        r'((?:\s*<p>.*?</p>\s*)*)'
        r'(\s*<hr\s*/>)',
        re.DOTALL,
    )
    def rep(m):
        lines = re.findall(r'<p>(.*?)</p>', m.group(2), re.DOTALL)
        meta_rows = []
        meta_items = []
        for line in lines:
            meta_items.extend(
                part.strip()
                for part in re.split(r'<br\s*/?>', line)
                if part.strip()
            )
        for line in meta_items:
            label_match = re.match(r'<strong>([^<]+)</strong>\s*(.*)', line, re.DOTALL)
            if label_match:
                meta_rows.append(
                    '<div class="cover-meta-row">'
                    f'<span>{label_match.group(1).rstrip(":")}</span>'
                    f'<strong>{label_match.group(2)}</strong>'
                    '</div>'
                )
            else:
                meta_rows.append(f'<div class="cover-meta-row"><strong>{line}</strong></div>')
        meta = f'<div class="cover-meta">{"".join(meta_rows)}</div>' if meta_rows else ''
        return (
            '<section class="cover-page">'
            '<div class="cover-shell">'
            '<div class="cover-topline">'
            '<span>M1 CHPS</span>'
            '<span>TP TM · 2025–2026</span>'
            '</div>'
            '<div class="cover-main">'
            '<p class="cover-label">Finite Difference Methods</p>'
            f'{m.group(1)}'
            '<p class="cover-subtitle">Analytical and numerical study of a linear reaction-diffusion equation</p>'
            '</div>'
            '<div class="cover-bottom">'
            f'{meta}'
            '<div class="cover-footer">Python / NumPy · Reaction-Diffusion Equation</div>'
            '</div>'
            '</div>'
            '</section>'
        )
    return pat.sub(rep, html, count=1)


def _wrap_toc(html: str) -> str:
    return re.sub(
        r'(<h2>Table of Contents</h2>)([\s\S]*?)(?=<h2>Abstract</h2>)',
        lambda m: f'<section class="toc-page">{m.group(1)}{m.group(2).strip()}</section>\n\n',
        html,
        count=1,
    )


def _wrap_abstract(html: str) -> str:
    return re.sub(
        r'(<h2>Abstract</h2>)([\s\S]*?)(?=<h2)',
        lambda m: m.group(1) + f'<div class="abstract">{m.group(2).strip()}</div>\n\n',
        html,
        count=1,
    )


_RE_THEAD = re.compile(r'<thead>[\s\S]*?</thead>', re.DOTALL)
_RE_SVG_URI = re.compile(r'(src="data:image/svg\+xml;base64,)([^"]+)(")')


def _recolor_svg(svg: str, color: str) -> str:
    """Make inherited-color SVG glyphs render in a concrete color."""
    def add_root_color(m):
        attrs = m.group(1)
        if re.search(r'\sfill=', attrs):
            return m.group(0)
        return f'<svg{attrs} fill="{color}" color="{color}">'

    svg = re.sub(r'<svg\b([^>]*)>', add_root_color, svg, count=1)
    # Some SVGs contain explicit black fills/strokes; keep "none" untouched.
    svg = re.sub(r'(?i)(fill:\s*)#(?:000|000000)\b', rf'\1{color}', svg)
    svg = re.sub(r'(?i)(stroke:\s*)#(?:000|000000)\b', rf'\1{color}', svg)
    svg = re.sub(r'(?i)(\sfill=")#(?:000|000000)(")', rf'\1{color}\2', svg)
    svg = re.sub(r'(?i)(\sstroke=")#(?:000|000000)(")', rf'\1{color}\2', svg)
    return svg


def _recolor_svg_uris(html: str, color: str) -> str:
    def rep(m):
        try:
            svg = base64.b64decode(m.group(2)).decode('utf-8')
            svg = _recolor_svg(svg, color)
            b64 = base64.b64encode(svg.encode('utf-8')).decode('ascii')
            return f'{m.group(1)}{b64}{m.group(3)}'
        except Exception:
            return m.group(0)

    return _RE_SVG_URI.sub(rep, html)


def _recolor_table_header_math(html: str) -> str:
    """WeasyPrint does not reliably apply CSS filters to SVG data URIs."""
    return _RE_THEAD.sub(lambda m: _recolor_svg_uris(m.group(0), '#ffffff'), html)


# ── 6.  CSS ────────────────────────────────────────────────────────────────────

CSS = r"""
/* ── Page ──────────────────────────────────────────────────────────────── */
@page {
    size: A4;
    margin: 2.4cm 2.7cm 2.7cm 2.7cm;
    @bottom-center {
        content: "— " counter(page) " —";
        font-family: Georgia, serif;
        font-size: 9pt;
        color: #777;
    }
}
@page :first { @bottom-center { content: ""; } }

/* ── Reset ──────────────────────────────────────────────────────────────── */
* { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Body ───────────────────────────────────────────────────────────────── */
body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.62;
    color: #1c1c1c;
    text-align: justify;
}

.cover-page {
    min-height: 225mm;
    display: flex;
    align-items: center;
    justify-content: center;
    break-after: page;
    page-break-after: always;
}
.cover-shell {
    position: relative;
    width: 100%;
    min-height: 184mm;
    padding: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    text-align: left;
}
.cover-topline {
    display: flex;
    justify-content: space-between;
    padding-bottom: 6mm;
    border-bottom: 1.1pt solid #14304f;
    color: #14304f;
    font-size: 9.5pt;
    font-weight: bold;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.cover-main {
    padding-top: 28mm;
    max-width: 162mm;
}
.cover-label {
    color: #2c6fad;
    font-size: 10pt;
    font-weight: bold;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    margin-bottom: 8mm;
}
.cover-shell h1 {
    font-size: 30pt;
    font-weight: bold;
    color: #14304f;
    line-height: 1.14;
    margin: 0 0 9mm;
    max-width: 158mm;
}
.cover-subtitle {
    max-width: 118mm;
    margin: 0;
    color: #3d4a58;
    font-size: 12.2pt;
    line-height: 1.5;
    text-align: left;
}
.cover-meta {
    width: 100%;
    border-top: 1.1pt solid #d6dee7;
    padding-top: 8mm;
}
.cover-meta-row {
    display: flex;
    align-items: baseline;
    padding: 2.2mm 0;
    border-bottom: 0.45pt solid #e5ebf1;
    font-size: 10pt;
    color: #27384c;
}
.cover-meta-row span {
    width: 31mm;
    flex: 0 0 31mm;
    margin-right: 8mm;
    color: #6a7785;
    font-size: 8.6pt;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.cover-meta-row strong {
    font-weight: normal;
}
.cover-footer {
    margin-top: 7mm;
    color: #6a7785;
    font-size: 8.8pt;
    letter-spacing: 0.06em;
    text-align: right;
    text-transform: uppercase;
}

/* ── Table of contents page ─────────────────────────────────────────────── */
.toc-page {
    min-height: 225mm;
    break-after: page;
    page-break-after: always;
}
.toc-page h2 {
    margin-top: 0;
}
.toc-page table {
    width: 100%;
    margin-top: 9pt;
    font-size: 7.5pt;
}
.toc-page td:first-child,
.toc-page th:first-child {
    width: 13%;
}
.toc-page td:last-child,
.toc-page th:last-child {
    width: 10%;
}
.toc-page th {
    padding: 3pt 6pt;
    font-size: 7.5pt;
}
.toc-page td {
    padding: 1.6pt 6pt;
    line-height: 1.12;
}

/* ── Abstract ───────────────────────────────────────────────────────────── */
.abstract {
    margin: 0 1.4cm 18pt;
    padding: 8pt 12pt;
    font-size: 9.5pt;
    line-height: 1.55;
    border-left: 3pt solid #2c6fad;
    background: #f6f9fd;
    color: #333;
}

/* ── Headings ───────────────────────────────────────────────────────────── */
h2 {
    font-size: 13pt;
    font-weight: bold;
    color: #14304f;
    margin: 22pt 0 7pt;
    padding-bottom: 3pt;
    border-bottom: 1.5pt solid #2c6fad;
    page-break-after: avoid;
}
h3 {
    font-size: 11pt;
    font-weight: bold;
    color: #1a3a5c;
    margin: 14pt 0 5pt;
    page-break-after: avoid;
}
h4 {
    font-size: 10.5pt;
    font-weight: bold;
    font-style: italic;
    color: #2c3e50;
    margin: 10pt 0 4pt;
    page-break-after: avoid;
}

/* ── Paragraphs ─────────────────────────────────────────────────────────── */
p { margin: 0 0 7pt; }

/* ── Inline math images ─────────────────────────────────────────────────── */
img.math-img-inline {
    vertical-align: -0.28em;
    max-height: 2.2em;
}

/* ── Display math ───────────────────────────────────────────────────────── */
.eq-block {
    text-align: center;
    margin: 10pt 0 12pt;
}
.eq-block img.math-img-display {
    max-width: 90%;
}
.eq-row {
    display: flex;
    align-items: center;
    margin: 10pt 0 12pt;
}
.eq-math {
    flex: 1;
    text-align: center;
}
.eq-math img.math-img-display {
    max-width: 100%;
}
.eq-num {
    min-width: 3.5em;
    text-align: right;
    font-size: 9.5pt;
    color: #555;
    padding-left: 6pt;
    font-family: Georgia, serif;
}

/* ── Boxed equations ────────────────────────────────────────────────────── */
.eq-boxed {
    background: #eef4fc;
    border-radius: 3pt;
    padding: 4pt 10pt;
    margin: 10pt auto;
    display: inline-flex;
    width: auto;
}
.eq-boxed .eq-math { flex: 1; }

/* ── Matrix (pmatrix HTML rendering) ───────────────────────────────────── */
.mx-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 4pt 0;
}
.mx-wrap {
    display: inline-flex;
    align-items: stretch;
    margin: 0 2pt;
}
.mx-lbr {
    width: 7pt;
    margin-right: 3pt;
    flex-shrink: 0;
    align-self: stretch;
    background:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 8 100' preserveAspectRatio='none'%3E%3Cpath d='M6.5 1H2V99H6.5' fill='none' stroke='%231c1c1c' stroke-width='1.5' stroke-linecap='square' stroke-linejoin='miter'/%3E%3C/svg%3E")
        center / 100% 100% no-repeat;
}
.mx-rbr {
    width: 7pt;
    margin-left: 3pt;
    flex-shrink: 0;
    align-self: stretch;
    background:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 8 100' preserveAspectRatio='none'%3E%3Cpath d='M1.5 1H6V99H1.5' fill='none' stroke='%231c1c1c' stroke-width='1.5' stroke-linecap='square' stroke-linejoin='miter'/%3E%3C/svg%3E")
        center / 100% 100% no-repeat;
}
.mx-tbl {
    border-collapse: separate;
    border-spacing: 0;
    margin: 0;
    font-size: inherit;
    border: none !important;
}
.mx-tbl tr, .mx-tbl td, .mx-tbl th {
    border: none !important;
    background: white !important;
}
.mx-tbl tbody tr:last-child td { border: none !important; }
.mx-cell {
    padding: 3pt 8pt;
    text-align: center;
    border: none !important;
    background: white !important;
    vertical-align: middle;
}
.mx-eq { font-size: 10.5pt; }

/* ── Tables ─────────────────────────────────────────────────────────────── */
table {
    border-collapse: collapse;
    width: auto;
    max-width: 100%;
    margin: 8pt auto 14pt;
    font-size: 9pt;
    page-break-inside: avoid;
}
thead tr             { background-color: #14304f; color: white; }
thead th img.math-img-inline { filter: brightness(0) invert(1); }
th {
    padding: 5pt 9pt;
    text-align: center;
    font-weight: bold;
    font-size: 8.5pt;
    letter-spacing: 0.02em;
}
td {
    padding: 4pt 9pt;
    text-align: center;
    border-bottom: 0.5pt solid #c8d4e0;
    vertical-align: middle;
}
tbody tr:nth-child(even) { background-color: #f2f6fb; }
tbody tr:last-child td   { border-bottom: 1pt solid #2c6fad; }

/* ── Code ───────────────────────────────────────────────────────────────── */
code {
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    background: #f3f3f3;
    padding: 1pt 3pt;
    border-radius: 2pt;
    color: #c0392b;
}
pre {
    background: #f8f8f8;
    border: 0.5pt solid #ddd;
    border-left: 3pt solid #2c6fad;
    padding: 7pt 10pt;
    font-size: 8pt;
    line-height: 1.45;
    margin: 8pt 0 12pt;
    page-break-inside: avoid;
    white-space: pre-wrap;
    word-break: break-all;
}
pre code { background: none; color: #1c1c1c; padding: 0; font-size: inherit; }

/* ── Figures ────────────────────────────────────────────────────────────── */
p img:not(.math-img-inline):not(.math-img-display) {
    display: block;
    max-width: 100%;
    height: auto;
    margin: 8pt auto 12pt;
    page-break-inside: avoid;
}

/* ── Lists ──────────────────────────────────────────────────────────────── */
ul, ol  { margin: 4pt 0 8pt 20pt; }
li      { margin: 3pt 0; }
li p    { margin: 0; }

/* ── Horizontal rule ────────────────────────────────────────────────────── */
hr { border: none; border-top: 0.8pt solid #ccd6e0; margin: 16pt 0; }

/* ── Emphasis ───────────────────────────────────────────────────────────── */
strong { color: #14304f; }

/* ── Math error fallback ────────────────────────────────────────────────── */
.math-err-d, .math-err-i {
    font-family: "Courier New", monospace;
    font-size: 8pt;
    color: #c0392b;
    background: #fff0f0;
    padding: 1pt 3pt;
}
"""

# ── 7.  HTML template ─────────────────────────────────────────────────────────

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""

# ── 8.  Main ──────────────────────────────────────────────────────────────────

def main():
    src = (HERE / "report.md").read_text(encoding="utf-8")

    print("  [1/4] Pre-processing math (rendering SVGs)…")
    preprocessed = preprocess(src)
    n_display = preprocessed.count('class="eq-block') + preprocessed.count('class="eq-row')
    n_inline  = preprocessed.count('math-img-inline')
    print(f"        {n_display} display blocks, {n_inline} inline images rendered.")

    print("  [2/4] Converting Markdown → HTML…")
    body = to_html(preprocessed)

    print("  [3/4] Post-processing layout…")
    body = _wrap_title_block(body)
    body = _wrap_toc(body)
    body = _wrap_abstract(body)
    body = _recolor_table_header_math(body)

    full_html = _TEMPLATE.format(css=CSS, body=body)

    out = HERE / "report.pdf"
    print(f"  [4/4] Rendering PDF → {out.name}…")
    from weasyprint import HTML as WP_HTML
    WP_HTML(string=full_html, base_url=str(HERE)).write_pdf(str(out))
    print(f"\n  Done.  {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
