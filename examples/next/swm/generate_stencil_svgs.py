#!/usr/bin/env python3
"""
Generate SVG stencil diagrams for the Shallow Water Model (SWM) operators.

Illustrates the Arakawa C-grid staggering and the stencil patterns for:
  Phase 1 — intermediate quantities: cu, cv, z, h
  Phase 2 — time-step updates:       ũ, ṽ, p̃
  Composite — full dependency from initial (p, u, v) to final (ũ, ṽ, p̃)

Shapes encode grid location — matching WHERE on the C-grid they live:
  ● circle (filled)  → vertex        (z)       — sits at grid intersections
  ━ horizontal bar   → x-edge        (u, cu)   — straddles vertical cell boundary
  ┃ vertical bar     → y-edge        (v, cv)   — straddles horizontal cell boundary
  (text only)        → cell center   (p, h)    — floats inside the cell

The background grid shows cell boundaries (dashed lines). Vertices sit at
intersections, edges sit on boundaries, and cell-center quantities sit
between boundaries — making the staggering immediately visible.

Output SVGs use the CSCS Reveal.js color palette.
"""

import argparse
import math
import os

# ─── Configuration ────────────────────────────────────────────────────────────

CELL = 65          # px between adjacent staggered positions (= half a cell)
CIRCLE_R = 14      # vertex circle radius
BAR_LONG = 42      # edge bar long axis (px) — elongated along edge direction
BAR_SHORT = 14     # edge bar short axis (px)
BAR_RX = 3         # edge bar corner radius
FONT = 12          # label font size
FONT_BAR = 11      # label font inside bars
FONT_CENTER = 13   # label font for cell-center text
STROKE = 1.6
STROKE_OUT = 2.2
ARROW_W = 1.1
PAD = 52           # padding around each sub-diagram
TITLE_H = 24       # space for title above diagram
ANIM_DT = 0.13     # seconds between animation steps (configurable via --anim-dt)
GAP = 16           # gap between sub-diagrams in combined SVGs

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..', 'cscs_revealjs_template', 'svg'
)

# CSCS palette — variable colors
VC = {
    'p': '#1F407A',   # dark blue  — cell center
    'u': '#007A96',   # teal       — x-edge
    'v': '#A60B16',   # dark red   — y-edge
    'z': '#72791C',   # green      — vertex
    'h': '#800080',   # purple     — Bernoulli (cell center)
}
ARROW_C = '#b5b5b5'
ARROW_THIN_C = '#cdcdcd'  # lighter arrows for phase-1 in composites
GRID_C = '#d0d3d8'
TEXT_C = '#2c2c2c'
TEXT_MUTED = '#888888'

# Shape type per variable — determines visual representation
SHAPE_TYPE = {
    'p': 'text',       # cell center: text only
    'u': 'bar_x',      # x-edge: horizontal bar
    'v': 'bar_y',      # y-edge: vertical bar
    'z': 'circle',     # vertex: circle
    'h': 'text',       # Bernoulli at cell center: text only
}

# Grid parity: whether cell boundaries pass through even or odd CELL offsets
# (0 = boundaries at 0, ±2, ±4 CELL; 1 = boundaries at ±1, ±3, ±5 CELL)
GRID_PARITY = {
    'z': (0, 0),   # vertex sits at boundary intersection
    'p': (1, 1),   # cell center sits between boundaries
    'h': (1, 1),
    'u': (0, 1),   # x-edge on vertical boundary, between horizontal ones
    'v': (1, 0),   # y-edge on horizontal boundary, between vertical ones
}


# ─── Stencil definitions ─────────────────────────────────────────────────────
# Each input: (dx, dy, label, var_type)
# dx, dy in CELL units relative to output at (0, 0)

STENCILS = {
    # Phase 1: intermediate quantities
    'cu': {
        'formula': 'cu = ⟨p⟩ₓ · u',
        'out_label': 'cu', 'out_var': 'u',
        'inputs': [
            (-1, 0, 'p', 'p'),
            ( 1, 0, 'p', 'p'),
        ],
    },
    'cv': {
        'formula': 'cv = ⟨p⟩ᵧ · v',
        'out_label': 'cv', 'out_var': 'v',
        'inputs': [
            (0, -1, 'p', 'p'),
            (0,  1, 'p', 'p'),
        ],
    },
    'z': {
        'formula': 'z (vorticity)',
        'out_label': 'z', 'out_var': 'z',
        'inputs': [
            (-1, -1, 'p', 'p'), ( 1, -1, 'p', 'p'),
            (-1,  1, 'p', 'p'), ( 1,  1, 'p', 'p'),
            ( 0, -1, 'u', 'u'), ( 0,  1, 'u', 'u'),
            (-1,  0, 'v', 'v'), ( 1,  0, 'v', 'v'),
        ],
    },
    'h': {
        'formula': 'h (Bernoulli)',
        'out_label': 'h', 'out_var': 'h',
        'inputs': [
            (-1, 0, 'u', 'u'), (1, 0, 'u', 'u'),
            (0, -1, 'v', 'v'), (0, 1, 'v', 'v'),
        ],
    },
    # Phase 2: time-step updates
    'unew': {
        'formula': 'u_new (u update)',
        'out_label': 'u', 'out_sub': 'new', 'out_var': 'u',
        'inputs': [
            ( 0, -1, 'z', 'z'), ( 0,  1, 'z', 'z'),
            (-1,  0, 'h', 'h'), ( 1,  0, 'h', 'h'),
            (-1, -1, 'V', 'v'), ( 1, -1, 'V', 'v'),
            (-1,  1, 'V', 'v'), ( 1,  1, 'V', 'v'),
        ],
    },
    'vnew': {
        'formula': 'v_new (v update)',
        'out_label': 'v', 'out_sub': 'new', 'out_var': 'v',
        'inputs': [
            (-1,  0, 'z', 'z'), ( 1,  0, 'z', 'z'),
            ( 0, -1, 'h', 'h'), ( 0,  1, 'h', 'h'),
            (-1, -1, 'U', 'u'), ( 1, -1, 'U', 'u'),
            (-1,  1, 'U', 'u'), ( 1,  1, 'U', 'u'),
        ],
    },
    'pnew': {
        'formula': 'p_new (p update)',
        'out_label': 'p', 'out_sub': 'new', 'out_var': 'p',
        'inputs': [
            (-1, 0, 'U', 'u'), (1, 0, 'U', 'u'),
            (0, -1, 'V', 'v'), (0, 1, 'V', 'v'),
        ],
    },
    # ── Composite stencils: full dependency chain with intermediates ──
    # Two-phase arrows: initial→intermediate (thin), intermediate→output (normal)
    # Animation: phase1 inputs → intermediates appear → phase2 arrows + output
    'u_composite': {
        'formula': 'u_new from (p, u, v)',
        'out_label': 'u', 'out_sub': 'new', 'out_var': 'u',
        'composite': True,
        # Phase 1: initial inputs → intermediate quantities
        'phase1': [
            # z at (0,-1): inputs from p(-1,-2), p(1,-2), u(0,-2), v(-1,-1), v(1,-1)
            {'name': 'z', 'pos': (0, -1), 'label': 'z', 'var': 'z',
             'inputs': [(-1, -2, 'p', 'p'), (1, -2, 'p', 'p'), (0, -2, 'u', 'u'),
                        (-1, -1, 'v', 'v'), (1, -1, 'v', 'v')]},
            # z at (0,1): inputs from p(-1,2), p(1,2), u(0,2), v(-1,1), v(1,1)
            {'name': 'z', 'pos': (0, 1), 'label': 'z', 'var': 'z',
             'inputs': [(-1, 2, 'p', 'p'), (1, 2, 'p', 'p'), (0, 2, 'u', 'u'),
                        (-1, 1, 'v', 'v'), (1, 1, 'v', 'v')]},
            # h at (-1,0): inputs from u(-2,0), u(0,0_skip), v(-1,-1), v(-1,1)
            {'name': 'h', 'pos': (-1, 0), 'label': 'h', 'var': 'h',
             'inputs': [(-2, 0, 'u', 'u'), (-1, -1, 'v', 'v'), (-1, 1, 'v', 'v')]},
            # h at (1,0): inputs from u(2,0), u(0,0_skip), v(1,-1), v(1,1)
            {'name': 'h', 'pos': (1, 0), 'label': 'h', 'var': 'h',
             'inputs': [(2, 0, 'u', 'u'), (1, -1, 'v', 'v'), (1, 1, 'v', 'v')]},
        ],
        # Phase 2: intermediates → output (these are the arrows from intermediates to output)
        'phase2_inputs': [
            (0, -1, 'z', 'z'), (0, 1, 'z', 'z'),
            (-1, 0, 'h', 'h'), (1, 0, 'h', 'h'),
        ],
    },
    'v_composite': {
        'formula': 'v_new from (p, u, v)',
        'out_label': 'v', 'out_sub': 'new', 'out_var': 'v',
        'composite': True,
        'phase1': [
            # z at (-1,0)
            {'name': 'z', 'pos': (-1, 0), 'label': 'z', 'var': 'z',
             'inputs': [(-2, -1, 'p', 'p'), (-2, 1, 'p', 'p'), (-1, -1, 'u', 'u'),
                        (-1, 1, 'u', 'u'), (-2, 0, 'v', 'v')]},
            # z at (1,0)
            {'name': 'z', 'pos': (1, 0), 'label': 'z', 'var': 'z',
             'inputs': [(2, -1, 'p', 'p'), (2, 1, 'p', 'p'), (1, -1, 'u', 'u'),
                        (1, 1, 'u', 'u'), (2, 0, 'v', 'v')]},
            # h at (0,-1)
            {'name': 'h', 'pos': (0, -1), 'label': 'h', 'var': 'h',
             'inputs': [(-1, -1, 'u', 'u'), (1, -1, 'u', 'u'), (0, -2, 'v', 'v')]},
            # h at (0,1)
            {'name': 'h', 'pos': (0, 1), 'label': 'h', 'var': 'h',
             'inputs': [(-1, 1, 'u', 'u'), (1, 1, 'u', 'u'), (0, 2, 'v', 'v')]},
        ],
        'phase2_inputs': [
            (-1, 0, 'z', 'z'), (1, 0, 'z', 'z'),
            (0, -1, 'h', 'h'), (0, 1, 'h', 'h'),
        ],
    },
    'p_composite': {
        'formula': 'p_new from (p, u, v)',
        'out_label': 'p', 'out_sub': 'new', 'out_var': 'p',
        'composite': True,
        'phase1': [
            # cu at (-1,0): cu = <p>_x * u, needs p(-2,0) and u(-1,0) [co-located]
            {'name': 'cu', 'pos': (-1, 0), 'label': 'cu', 'var': 'u',
             'inputs': [(-2, 0, 'p', 'p')]},
            # cu at (1,0)
            {'name': 'cu', 'pos': (1, 0), 'label': 'cu', 'var': 'u',
             'inputs': [(2, 0, 'p', 'p')]},
            # cv at (0,-1): cv = <p>_y * v, needs p(0,-2) and v(0,-1) [co-located]
            {'name': 'cv', 'pos': (0, -1), 'label': 'cv', 'var': 'v',
             'inputs': [(0, -2, 'p', 'p')]},
            # cv at (0,1)
            {'name': 'cv', 'pos': (0, 1), 'label': 'cv', 'var': 'v',
             'inputs': [(0, 2, 'p', 'p')]},
        ],
        'phase2_inputs': [
            (-1, 0, 'cu', 'u'), (1, 0, 'cu', 'u'),
            (0, -1, 'cv', 'v'), (0, 1, 'cv', 'v'),
        ],
    },
}


# ─── SVG element helpers ─────────────────────────────────────────────────────

def _shape_margin(var, dx, dy):
    """Compute arrow margin from a shape's center in direction (dx, dy).

    Uses line–bounding-box intersection so arrows stop at the shape edge.
    """
    d = math.hypot(dx, dy)
    if d < 0.01:
        return 0
    ndx, ndy = abs(dx / d), abs(dy / d)

    stype = SHAPE_TYPE.get(var, 'text')
    if stype == 'circle':
        return CIRCLE_R + 3
    elif stype == 'bar_x':
        hw, hh = BAR_LONG / 2, BAR_SHORT / 2
    elif stype == 'bar_y':
        hw, hh = BAR_SHORT / 2, BAR_LONG / 2
    else:  # text
        hw, hh = 10, 8

    # Line–rect intersection distance from center
    if ndx < 0.01:
        return hh + 3
    if ndy < 0.01:
        return hw + 3
    return min(hw / ndx, hh / ndy) + 3


def _label_tspan(label, sub=None, fill_color='#ffffff', font_size=FONT, fw='500'):
    """Return SVG text content, optionally with a subscript."""
    if sub:
        sub_size = max(7, int(font_size * 0.6))
        return (f'{label}<tspan font-size="{sub_size}" '
                f'baseline-shift="sub">{sub}</tspan>')
    return label


def shape_svg(cx, cy, label, var, is_output=False, cls='', sub=None):
    """Generate SVG group for a shape with label and optional subscript."""
    color = VC.get(var, TEXT_C)
    stype = SHAPE_TYPE.get(var, 'text')
    sw = STROKE_OUT if is_output else STROKE
    fw = '700' if is_output else '500'
    cls_attr = f' class="{cls}"' if cls else ''

    s = f'<g{cls_attr}>\n'

    if stype == 'circle':
        r = CIRCLE_R
        opacity = '0.82' if is_output else '0.62'
        s += f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>\n'
        text_content = _label_tspan(label, sub, '#ffffff', FONT, fw)
        s += (f'  <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
              f'font-family="Arial,sans-serif" font-size="{FONT}" font-weight="{fw}" '
              f'fill="#ffffff">{text_content}</text>\n')

    elif stype in ('bar_x', 'bar_y'):
        if stype == 'bar_x':
            bw, bh = BAR_LONG, BAR_SHORT
        else:
            bw, bh = BAR_SHORT, BAR_LONG
        rx = BAR_RX
        opacity = '0.82' if is_output else '0.62'
        s += (f'  <rect x="{cx - bw/2}" y="{cy - bh/2}" width="{bw}" height="{bh}" '
              f'rx="{rx}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>\n')
        text_content = _label_tspan(label, sub, '#ffffff', FONT_BAR, fw)
        s += (f'  <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
              f'font-family="Arial,sans-serif" font-size="{FONT_BAR}" font-weight="{fw}" '
              f'fill="#ffffff">{text_content}</text>\n')

    else:  # text — cell center, no shape
        text_content = _label_tspan(label, sub, color, FONT_CENTER, fw)
        s += (f'  <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
              f'font-family="Arial,sans-serif" font-size="{FONT_CENTER}" font-weight="{fw}" '
              f'fill="{color}">{text_content}</text>\n')

    s += '</g>\n'
    return s


def arrow_svg(x1, y1, x2, y2, var_from, var_to, cls='', thin=False):
    """Generate SVG arrow line from (x1,y1) toward (x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d < 1:
        return ''
    m1 = _shape_margin(var_from, dx, dy)
    m2 = _shape_margin(var_to, -dx, -dy)
    ax1 = x1 + dx * (m1 / d)
    ay1 = y1 + dy * (m1 / d)
    ax2 = x2 - dx * (m2 / d)
    ay2 = y2 - dy * (m2 / d)
    cls_attr = f' class="{cls}"' if cls else ''
    color = ARROW_THIN_C if thin else ARROW_C
    width = ARROW_W * 0.7 if thin else ARROW_W
    marker = 'url(#ah-thin)' if thin else 'url(#ah)'
    return (f'<line{cls_attr} x1="{ax1:.1f}" y1="{ay1:.1f}" '
            f'x2="{ax2:.1f}" y2="{ay2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" marker-end="{marker}"/>\n')


def grid_lines_svg(cx, cy, out_var, inputs):
    """Draw cell boundary grid lines. Only boundaries, not through cell centers."""
    xp, yp = GRID_PARITY.get(out_var, (1, 1))

    # Determine extent from inputs
    max_x = max((abs(inp[0]) for inp in inputs), default=1)
    max_y = max((abs(inp[1]) for inp in inputs), default=1)
    margin = 18

    lines = ''
    # Vertical cell boundaries
    for n in range(-6, 7):
        if abs(n) > max_x + 1.5:
            continue
        if n % 2 == xp:  # matches the parity for cell boundaries
            x = cx + n * CELL
            y_lo = cy - max_y * CELL - margin
            y_hi = cy + max_y * CELL + margin
            lines += (f'<line x1="{x}" y1="{y_lo}" x2="{x}" y2="{y_hi}" '
                      f'stroke="{GRID_C}" stroke-width="0.9" stroke-dasharray="4,3"/>\n')

    # Horizontal cell boundaries
    for n in range(-6, 7):
        if abs(n) > max_y + 1.5:
            continue
        if n % 2 == yp:
            y = cy + n * CELL
            x_lo = cx - max_x * CELL - margin
            x_hi = cx + max_x * CELL + margin
            lines += (f'<line x1="{x_lo}" y1="{y}" x2="{x_hi}" y2="{y}" '
                      f'stroke="{GRID_C}" stroke-width="0.9" stroke-dasharray="4,3"/>\n')

    return lines


# ─── Stencil diagram generation ──────────────────────────────────────────────

def _collect_all_positions(st):
    """Collect all input positions for extent calculation, including composite intermediates."""
    positions = []
    if st.get('composite'):
        for inter in st['phase1']:
            for inp in inter['inputs']:
                positions.append((inp[0], inp[1]))
            positions.append(inter['pos'])
        for inp in st['phase2_inputs']:
            positions.append((inp[0], inp[1]))
    else:
        for inp in st['inputs']:
            positions.append((inp[0], inp[1]))
    return positions


def render_stencil(name, ox=0, oy=0, animate=True, id_prefix=''):
    """Render a stencil diagram as an SVG <g> group.

    Returns (svg_string, width, height, num_anim_steps).
    """
    st = STENCILS[name]
    is_composite = st.get('composite', False)

    # Compute extent dynamically from all positions
    all_pos = _collect_all_positions(st)
    extent_x = max((abs(p[0]) for p in all_pos), default=1)
    extent_y = max((abs(p[1]) for p in all_pos), default=1)
    extent_x = max(extent_x, 1)
    extent_y = max(extent_y, 1)

    # Sub-diagram dimensions
    w = 2 * PAD + 2 * extent_x * CELL
    h = 2 * PAD + 2 * extent_y * CELL + TITLE_H

    # Center of the stencil pattern
    cx = PAD + extent_x * CELL
    cy = PAD + extent_y * CELL + TITLE_H

    pfx = id_prefix or name
    svg = f'<g transform="translate({ox},{oy})" id="stencil-{pfx}">\n'

    # Background grid — cell boundaries only
    grid_inputs = [(p[0], p[1], '', '') for p in all_pos]
    svg += grid_lines_svg(cx, cy, st['out_var'], grid_inputs)

    # Title
    svg += (f'<text x="{w / 2}" y="{TITLE_H - 4}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="13" font-weight="600" '
            f'fill="{TEXT_MUTED}">{st["formula"]}</text>\n')

    out_sub = st.get('out_sub')
    step = 0

    if is_composite:
        # ── Phase 1: thin arrows from initial inputs → intermediate shapes ──
        # Deduplicate initial inputs (same position may feed multiple intermediates)
        seen_inputs = {}  # (dx,dy) → anim step
        for inter in st['phase1']:
            for inp in inter['inputs']:
                key = (inp[0], inp[1])
                if key not in seen_inputs:
                    seen_inputs[key] = step
                    ix = cx + inp[0] * CELL
                    iy = cy + inp[1] * CELL
                    # Arrow to this intermediate
                    tgt_x = cx + inter['pos'][0] * CELL
                    tgt_y = cy + inter['pos'][1] * CELL
                    cls = f'a-{pfx}-{step}' if animate else ''
                    svg += arrow_svg(ix, iy, tgt_x, tgt_y, inp[3], inter['var'],
                                     cls, thin=True)
                    step += 1
                else:
                    # Still draw the arrow but reuse the same anim class
                    ix = cx + inp[0] * CELL
                    iy = cy + inp[1] * CELL
                    tgt_x = cx + inter['pos'][0] * CELL
                    tgt_y = cy + inter['pos'][1] * CELL
                    existing_step = seen_inputs[key]
                    cls = f'a-{pfx}-{existing_step}' if animate else ''
                    svg += arrow_svg(ix, iy, tgt_x, tgt_y, inp[3], inter['var'],
                                     cls, thin=True)

        # Draw initial input shapes (on top of thin arrows)
        for key, s_idx in seen_inputs.items():
            ix = cx + key[0] * CELL
            iy = cy + key[1] * CELL
            # Find the var type for this input
            var_type = None
            for inter in st['phase1']:
                for inp in inter['inputs']:
                    if (inp[0], inp[1]) == key:
                        var_type = inp[3]
                        break
                if var_type:
                    break
            label = None
            for inter in st['phase1']:
                for inp in inter['inputs']:
                    if (inp[0], inp[1]) == key:
                        label = inp[2]
                        break
                if label:
                    break
            cls = f'a-{pfx}-{s_idx}' if animate else ''
            svg += shape_svg(ix, iy, label, var_type, cls=cls)

        # ── Intermediate shapes ──
        n_phase1 = step
        for inter in st['phase1']:
            tgt_x = cx + inter['pos'][0] * CELL
            tgt_y = cy + inter['pos'][1] * CELL
            cls = f'a-{pfx}-inter-{inter["pos"][0]}_{inter["pos"][1]}' if animate else ''
            svg += shape_svg(tgt_x, tgt_y, inter['label'], inter['var'], cls=cls)

        # ── Phase 2: normal arrows from intermediates → output ──
        for inp in st['phase2_inputs']:
            ix = cx + inp[0] * CELL
            iy = cy + inp[1] * CELL
            cls = f'a-{pfx}-p2-{step}' if animate else ''
            svg += arrow_svg(ix, iy, cx, cy, inp[3], st['out_var'], cls)
            step += 1

    else:
        inputs = st['inputs']
        # Arrows first (drawn behind shapes)
        for inp in inputs:
            ix = cx + inp[0] * CELL
            iy = cy + inp[1] * CELL
            cls = f'a-{pfx}-{step}' if animate else ''
            svg += arrow_svg(ix, iy, cx, cy, inp[3], st['out_var'], cls)
            step += 1

        # Input shapes (on top of arrows)
        step_s = 0
        for inp in inputs:
            ix = cx + inp[0] * CELL
            iy = cy + inp[1] * CELL
            cls = f'a-{pfx}-{step_s}' if animate else ''
            svg += shape_svg(ix, iy, inp[2], inp[3], cls=cls)
            step_s += 1

    # Output shape (drawn last, on top)
    cls = f'a-{pfx}-out' if animate else ''
    svg += shape_svg(cx, cy, st['out_label'], st['out_var'], is_output=True, cls=cls,
                     sub=out_sub)
    step += 1

    svg += '</g>\n'
    return svg, w, h, step


def animation_css(stencil_names, id_prefixes=None):
    """Generate CSS @keyframes for all stencils."""
    css = '@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }\n'
    css += '@keyframes popIn { 0% { opacity:0; transform:scale(0.7); } 100% { opacity:1; transform:scale(1); } }\n'

    if id_prefixes is None:
        id_prefixes = stencil_names

    for name, pfx in zip(stencil_names, id_prefixes):
        st = STENCILS[name]

        if st.get('composite'):
            # 3-phase animation for composite stencils:
            # Phase 1: initial inputs + thin arrows fade in
            # Phase 2: intermediate shapes pop in
            # Phase 3: phase2 arrows fade in + output pops

            # Count unique initial inputs for phase 1 timing
            seen = set()
            n_phase1 = 0
            for inter in st['phase1']:
                for inp in inter['inputs']:
                    key = (inp[0], inp[1])
                    if key not in seen:
                        seen.add(key)
                        n_phase1 += 1

            # Phase 1: initial inputs fade in
            for i in range(n_phase1):
                delay = i * ANIM_DT
                css += f'.a-{pfx}-{i} {{ opacity:0; animation: fadeIn 0.25s ease-out {delay:.2f}s both; }}\n'

            # Phase 2: intermediates pop in after all phase1 inputs
            inter_delay = n_phase1 * ANIM_DT + 0.15
            for inter in st['phase1']:
                pos_key = f'{inter["pos"][0]}_{inter["pos"][1]}'
                css += (f'.a-{pfx}-inter-{pos_key} {{ opacity:0; animation: popIn 0.3s ease-out {inter_delay:.2f}s both; '
                        f'transform-box: fill-box; transform-origin: center; }}\n')

            # Phase 3: phase2 arrows + output
            p2_start = inter_delay + 0.35
            step = n_phase1  # continuing step count from render_stencil
            for i, inp in enumerate(st['phase2_inputs']):
                delay = p2_start + i * ANIM_DT
                css += f'.a-{pfx}-p2-{step + i} {{ opacity:0; animation: fadeIn 0.25s ease-out {delay:.2f}s both; }}\n'

            out_delay = p2_start + len(st['phase2_inputs']) * ANIM_DT + 0.1
            css += (f'.a-{pfx}-out {{ opacity:0; animation: popIn 0.35s ease-out {out_delay:.2f}s both; '
                    f'transform-box: fill-box; transform-origin: center; }}\n')
        else:
            n_inputs = len(st['inputs'])

            for i in range(n_inputs):
                delay = i * ANIM_DT
                css += f'.a-{pfx}-{i} {{ opacity:0; animation: fadeIn 0.25s ease-out {delay:.2f}s both; }}\n'

            out_delay = n_inputs * ANIM_DT + 0.1
            css += (f'.a-{pfx}-out {{ opacity:0; animation: popIn 0.35s ease-out {out_delay:.2f}s both; '
                    f'transform-box: fill-box; transform-origin: center; }}\n')

    return css


def svg_defs():
    """Return common SVG <defs> (arrowhead markers)."""
    return f'''<defs>
  <marker id="ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L8,3 L0,6 L2,3 Z" fill="{ARROW_C}"/>
  </marker>
  <marker id="ah-thin" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L7,2.5 L0,5 L1.5,2.5 Z" fill="{ARROW_THIN_C}"/>
  </marker>
</defs>
'''


def wrap_svg(content, width, height, css=''):
    """Wrap content in a complete SVG document."""
    style = f'<style>\n{css}</style>\n' if css else ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">\n'
            f'{style}{svg_defs()}{content}</svg>\n')


# ─── Legend ───────────────────────────────────────────────────────────────────

def legend_svg(ox, oy, show_intermediates=False):
    """Draw a shape/color legend."""
    items = [
        ('text', 'p', 'p — cell center'),
        ('bar_x', 'u', 'u — x-edge'),
        ('bar_y', 'v', 'v — y-edge'),
        ('circle', 'z', 'z — vertex'),
    ]
    if show_intermediates:
        items.append(('text', 'h', 'h — cell center'))

    svg = f'<g transform="translate({ox},{oy})">\n'
    x = 0
    for stype, var, desc in items:
        color = VC[var]
        sc = 0.7
        if stype == 'circle':
            r = CIRCLE_R * sc
            svg += f'  <circle cx="{x + r}" cy="0" r="{r}" fill="{color}" fill-opacity="0.62" stroke="none"/>\n'
            svg += (f'  <text x="{x + 2*r + 6}" y="0" dominant-baseline="central" '
                    f'font-family="Arial,sans-serif" font-size="11" fill="{TEXT_MUTED}">{desc}</text>\n')
        elif stype == 'bar_x':
            bw, bh = BAR_LONG * sc, BAR_SHORT * sc
            svg += (f'  <rect x="{x}" y="{-bh/2}" width="{bw}" height="{bh}" rx="{BAR_RX*sc}" '
                    f'fill="{color}" fill-opacity="0.62" stroke="none"/>\n')
            svg += (f'  <text x="{x + bw + 6}" y="0" dominant-baseline="central" '
                    f'font-family="Arial,sans-serif" font-size="11" fill="{TEXT_MUTED}">{desc}</text>\n')
        elif stype == 'bar_y':
            bw, bh = BAR_SHORT * sc, BAR_LONG * sc
            svg += (f'  <rect x="{x}" y="{-bh/2}" width="{bw}" height="{bh}" rx="{BAR_RX*sc}" '
                    f'fill="{color}" fill-opacity="0.62" stroke="none"/>\n')
            svg += (f'  <text x="{x + bw + 6}" y="0" dominant-baseline="central" '
                    f'font-family="Arial,sans-serif" font-size="11" fill="{TEXT_MUTED}">{desc}</text>\n')
        else:  # text
            svg += (f'  <text x="{x}" y="0" dominant-baseline="central" '
                    f'font-family="Arial,sans-serif" font-size="12" font-weight="600" '
                    f'fill="{color}">{var}</text>\n')
            svg += (f'  <text x="{x + 14}" y="0" dominant-baseline="central" '
                    f'font-family="Arial,sans-serif" font-size="11" fill="{TEXT_MUTED}">— {desc.split("— ")[1]}</text>\n')
        x += 140
    svg += '</g>\n'
    return svg


# ─── Combined SVG generators ─────────────────────────────────────────────────

def _stencil_dims(name):
    """Return (width, height) for a stencil sub-diagram."""
    st = STENCILS[name]
    all_pos = _collect_all_positions(st)
    ex = max((abs(p[0]) for p in all_pos), default=1)
    ey = max((abs(p[1]) for p in all_pos), default=1)
    ex, ey = max(ex, 1), max(ey, 1)
    return 2 * PAD + 2 * ex * CELL, 2 * PAD + 2 * ey * CELL + TITLE_H


def generate_phase1_svg():
    """Generate combined SVG for Phase 1 intermediates (cu, cv, z, h)."""
    names = ['cu', 'cv', 'z', 'h']
    sub_w, sub_h = _stencil_dims(names[0])  # all same size

    cols, rows = 2, 2
    total_w = cols * sub_w + (cols - 1) * GAP
    total_h = rows * sub_h + (rows - 1) * GAP + 40

    content = ''
    for idx, name in enumerate(names):
        col = idx % cols
        row = idx // cols
        ox = col * (sub_w + GAP)
        oy = row * (sub_h + GAP)
        svg_part, _, _, _ = render_stencil(name, ox, oy)
        content += svg_part

    content += legend_svg(20, total_h - 20)

    css = animation_css(names)
    return wrap_svg(content, total_w, total_h, css)


def generate_phase2_svg():
    """Generate combined SVG for Phase 2 updates (ũ, ṽ, p̃)."""
    names = ['unew', 'vnew', 'pnew']
    sub_w, sub_h = _stencil_dims(names[0])

    cols = 3
    total_w = cols * sub_w + (cols - 1) * GAP
    total_h = sub_h + 40

    content = ''
    for idx, name in enumerate(names):
        ox = idx * (sub_w + GAP)
        svg_part, _, _, _ = render_stencil(name, ox, 0)
        content += svg_part

    content += legend_svg(20, total_h - 20, show_intermediates=True)

    css = animation_css(names)
    return wrap_svg(content, total_w, total_h, css)


def generate_composite_svg():
    """Generate combined SVG for composite stencils (ũ, ṽ, p̃ from initial)."""
    names = ['u_composite', 'v_composite', 'p_composite']
    sub_w, sub_h = _stencil_dims(names[0])  # all extent=2, same size

    cols = 3
    total_w = cols * sub_w + (cols - 1) * GAP
    total_h = sub_h + 40

    content = ''
    for idx, name in enumerate(names):
        ox = idx * (sub_w + GAP)
        svg_part, _, _, _ = render_stencil(name, ox, 0)
        content += svg_part

    content += legend_svg(20, total_h - 20)

    css = animation_css(names)
    return wrap_svg(content, total_w, total_h, css)


def generate_individual_svgs():
    """Generate individual SVG files for each stencil."""
    results = {}
    for name in STENCILS:
        svg_part, w, h, _ = render_stencil(name, 0, 0)
        css = animation_css([name])
        results[name] = wrap_svg(svg_part, w, h + 4, css)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global ANIM_DT

    parser = argparse.ArgumentParser(description='Generate SWM stencil SVGs')
    parser.add_argument('--anim-dt', type=float, default=ANIM_DT,
                        help=f'Animation step delay in seconds (default: {ANIM_DT})')
    args = parser.parse_args()
    ANIM_DT = args.anim_dt

    out_dir = os.path.normpath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    phase1 = generate_phase1_svg()
    phase1_path = os.path.join(out_dir, 'swm_intermediates.svg')
    with open(phase1_path, 'w') as f:
        f.write(phase1)
    print(f'  wrote {phase1_path}')

    phase2 = generate_phase2_svg()
    phase2_path = os.path.join(out_dir, 'swm_updates.svg')
    with open(phase2_path, 'w') as f:
        f.write(phase2)
    print(f'  wrote {phase2_path}')

    composite = generate_composite_svg()
    composite_path = os.path.join(out_dir, 'swm_composite.svg')
    with open(composite_path, 'w') as f:
        f.write(composite)
    print(f'  wrote {composite_path}')

    individuals = generate_individual_svgs()
    for name, svg in individuals.items():
        path = os.path.join(out_dir, f'swm_{name}.svg')
        with open(path, 'w') as f:
            f.write(svg)
        print(f'  wrote {path}')

    print(f'\nDone — {3 + len(individuals)} SVGs written to {out_dir}/')


if __name__ == '__main__':
    main()
