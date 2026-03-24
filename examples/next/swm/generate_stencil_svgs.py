#!/usr/bin/env python3
"""
Generate SVG stencil diagrams for the Shallow Water Model (SWM) operators.

Illustrates the Arakawa C-grid staggering and the stencil patterns for:
  Phase 1 — intermediate quantities: cu, cv, z, h
  Phase 2 — time-step updates:       ũ, ṽ, p̃
  Composite — full dependency from initial (p, u, v) to final (ũ, ṽ, p̃)

Shapes encode grid location — matching WHERE on the C-grid they live:
  ● circle (filled)  → vertex        (z)       — sits at grid intersections
  ┃ vertical bar     → x-edge        (u, cu)   — straddles vertical cell boundary
  ━ horizontal bar   → y-edge        (v, cv)   — straddles horizontal cell boundary
  (text only)        → cell center   (p, h)    — floats inside the cell

Uses the drawsvg library for SVG generation and CSCS Reveal.js color palette.
"""

import argparse
import math
import os

import drawsvg as dw

# ─── Layout constants ────────────────────────────────────────────────────────

CELL = 65           # px between adjacent staggered positions
CIRCLE_R = 14       # vertex circle radius
BAR = (42, 14)      # (long, short) edge bar dimensions
BAR_RX = 3          # edge bar corner radius
FONT = 12           # label font size (circle)
FONT_BAR = 11       # label font inside bars
FONT_CENTER = 13    # label font for cell-center text
ARROW_W = 1.1       # arrow stroke width
PAD = 52            # padding around each sub-diagram
TITLE_H = 24        # space for title above diagram
PHASE_PAUSE = 0.6   # seconds between animation phases
GAP = 16            # gap between sub-diagrams in combined SVGs

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..', 'cscs_revealjs_template', 'svg'
)

# ─── Colors ───────────────────────────────────────────────────────────────────

VAR_COLOR = {
    'p': '#1F407A', 'u': '#007A96', 'v': '#A60B16',
    'z': '#72791C', 'h': '#800080',
}
GRID_C = '#d0d3d8'
TEXT_MUTED = '#888888'

# ─── Grid staggering ─────────────────────────────────────────────────────────

# shape type per variable
SHAPE = {'p': 'text', 'u': 'bar_y', 'v': 'bar_x', 'z': 'circle', 'h': 'text'}

# Grid parity: cell boundaries at even (0) or odd (1) CELL multiples
GRID_PARITY = {
    'z': (0, 0), 'p': (1, 1), 'h': (1, 1), 'u': (0, 1), 'v': (1, 0),
}


# ─── Stencil definitions ─────────────────────────────────────────────────────
# Each input: (dx, dy, label, var_type)  — offsets in CELL units from output

STENCILS = {
    # Phase 1: intermediate quantities
    'cu': {
        'formula': 'U = ⟨p⟩ₓ · u', 'out_label': 'U', 'out_var': 'u',
        'inputs': [(-1, 0, 'p', 'p'), (1, 0, 'p', 'p')],
    },
    'cv': {
        'formula': 'V = ⟨p⟩ᵧ · v', 'out_label': 'V', 'out_var': 'v',
        'inputs': [(0, -1, 'p', 'p'), (0, 1, 'p', 'p')],
    },
    'z': {
        'formula': 'z (vorticity)', 'out_label': 'z', 'out_var': 'z',
        'inputs': [
            (-1, -1, 'p', 'p'), (1, -1, 'p', 'p'),
            (-1,  1, 'p', 'p'), (1,  1, 'p', 'p'),
            ( 0, -1, 'u', 'u'), (0,  1, 'u', 'u'),
            (-1,  0, 'v', 'v'), (1,  0, 'v', 'v'),
        ],
    },
    'h': {
        'formula': 'h (Bernoulli)', 'out_label': 'h', 'out_var': 'h',
        'inputs': [(-1, 0, 'u', 'u'), (1, 0, 'u', 'u'),
                   (0, -1, 'v', 'v'), (0, 1, 'v', 'v')],
    },
    # Phase 2: time-step updates
    'unew': {
        'formula': 'ũ (u update)', 'out_label': 'ũ', 'out_var': 'u',
        'inputs': [
            ( 0, -1, 'z', 'z'), ( 0,  1, 'z', 'z'),
            (-1,  0, 'h', 'h'), ( 1,  0, 'h', 'h'),
            (-1, -1, 'V', 'v'), ( 1, -1, 'V', 'v'),
            (-1,  1, 'V', 'v'), ( 1,  1, 'V', 'v'),
        ],
    },
    'vnew': {
        'formula': 'ṽ (v update)', 'out_label': 'ṽ', 'out_var': 'v',
        'inputs': [
            (-1,  0, 'z', 'z'), ( 1,  0, 'z', 'z'),
            ( 0, -1, 'h', 'h'), ( 0,  1, 'h', 'h'),
            (-1, -1, 'U', 'u'), ( 1, -1, 'U', 'u'),
            (-1,  1, 'U', 'u'), ( 1,  1, 'U', 'u'),
        ],
    },
    'pnew': {
        'formula': 'p\u0303 (p update)', 'out_label': 'p\u0303', 'out_var': 'p',
        'inputs': [(-1, 0, 'U', 'u'), (1, 0, 'U', 'u'),
                   (0, -1, 'V', 'v'), (0, 1, 'V', 'v')],
    },
    # ── Composite stencils: full dependency chain with intermediates ──
    'u_composite': {
        'formula': 'ũ from (p, u, v)', 'out_label': 'ũ', 'out_var': 'u',
        'composite': True,
        'phase1_groups': [
            [   # Group 0: z (vorticity)
                {'pos': (0, -1), 'label': 'z', 'var': 'z',
                 'inputs': [(-1, -2, 'p', 'p'), (1, -2, 'p', 'p'),
                            (-1, 0, 'p', 'p'), (1, 0, 'p', 'p'),
                            (0, -2, 'u', 'u'), (0, 0, 'u', 'u'),
                            (-1, -1, 'v', 'v'), (1, -1, 'v', 'v')]},
                {'pos': (0, 1), 'label': 'z', 'var': 'z',
                 'inputs': [(-1, 0, 'p', 'p'), (1, 0, 'p', 'p'),
                            (-1, 2, 'p', 'p'), (1, 2, 'p', 'p'),
                            (0, 0, 'u', 'u'), (0, 2, 'u', 'u'),
                            (-1, 1, 'v', 'v'), (1, 1, 'v', 'v')]},
            ],
            [   # Group 1: h (Bernoulli)
                {'pos': (-1, 0), 'label': 'h', 'var': 'h',
                 'inputs': [(-2, 0, 'u', 'u'), (0, 0, 'u', 'u'),
                            (-1, -1, 'v', 'v'), (-1, 1, 'v', 'v')]},
                {'pos': (1, 0), 'label': 'h', 'var': 'h',
                 'inputs': [(0, 0, 'u', 'u'), (2, 0, 'u', 'u'),
                            (1, -1, 'v', 'v'), (1, 1, 'v', 'v')]},
            ],
        ],
        'phase2_inputs': [(0, -1, 'z', 'z'), (0, 1, 'z', 'z'),
                          (-1, 0, 'h', 'h'), (1, 0, 'h', 'h')],
    },
    'v_composite': {
        'formula': 'ṽ from (p, u, v)', 'out_label': 'ṽ', 'out_var': 'v',
        'composite': True,
        'phase1_groups': [
            [   # Group 0: z (vorticity)
                {'pos': (-1, 0), 'label': 'z', 'var': 'z',
                 'inputs': [(-2, -1, 'p', 'p'), (-2, 1, 'p', 'p'),
                            (0, -1, 'p', 'p'), (0, 1, 'p', 'p'),
                            (-1, -1, 'u', 'u'), (-1, 1, 'u', 'u'),
                            (-2, 0, 'v', 'v'), (0, 0, 'v', 'v')]},
                {'pos': (1, 0), 'label': 'z', 'var': 'z',
                 'inputs': [(0, -1, 'p', 'p'), (0, 1, 'p', 'p'),
                            (2, -1, 'p', 'p'), (2, 1, 'p', 'p'),
                            (1, -1, 'u', 'u'), (1, 1, 'u', 'u'),
                            (0, 0, 'v', 'v'), (2, 0, 'v', 'v')]},
            ],
            [   # Group 1: h (Bernoulli)
                {'pos': (0, -1), 'label': 'h', 'var': 'h',
                 'inputs': [(-1, -1, 'u', 'u'), (1, -1, 'u', 'u'),
                            (0, -2, 'v', 'v'), (0, 0, 'v', 'v')]},
                {'pos': (0, 1), 'label': 'h', 'var': 'h',
                 'inputs': [(-1, 1, 'u', 'u'), (1, 1, 'u', 'u'),
                            (0, 0, 'v', 'v'), (0, 2, 'v', 'v')]},
            ],
        ],
        'phase2_inputs': [(-1, 0, 'z', 'z'), (1, 0, 'z', 'z'),
                          (0, -1, 'h', 'h'), (0, 1, 'h', 'h')],
    },
    'p_composite': {
        'formula': 'p\u0303 from (p, U, V)', 'out_label': 'p\u0303', 'out_var': 'p',
        'composite': True,
        'phase1_groups': [
            [   # Group 0: U and V (simultaneous)
                {'pos': (-1, 0), 'label': 'U', 'var': 'u',
                 'inputs': [(-2, 0, 'p', 'p'), (0, 0, 'p', 'p')]},
                {'pos': (1, 0), 'label': 'U', 'var': 'u',
                 'inputs': [(0, 0, 'p', 'p'), (2, 0, 'p', 'p')]},
                {'pos': (0, -1), 'label': 'V', 'var': 'v',
                 'inputs': [(0, -2, 'p', 'p'), (0, 0, 'p', 'p')]},
                {'pos': (0, 1), 'label': 'V', 'var': 'v',
                 'inputs': [(0, 0, 'p', 'p'), (0, 2, 'p', 'p')]},
            ],
        ],
        'phase2_inputs': [(-1, 0, 'U', 'u'), (1, 0, 'U', 'u'),
                          (0, -1, 'V', 'v'), (0, 1, 'V', 'v')],
    },
}


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def _shape_margin(var, dx, dy):
    """Distance from shape center to edge in direction (dx, dy)."""
    d = math.hypot(dx, dy)
    if d < 0.01:
        return 0
    ndx, ndy = abs(dx / d), abs(dy / d)
    stype = SHAPE.get(var, 'text')
    if stype == 'circle':
        return CIRCLE_R + 3
    hw, hh = (BAR[0] / 2, BAR[1] / 2) if stype == 'bar_x' else \
             (BAR[1] / 2, BAR[0] / 2) if stype == 'bar_y' else (10, 8)
    if ndx < 0.01:
        return hh + 3
    if ndy < 0.01:
        return hw + 3
    return min(hw / ndx, hh / ndy) + 3


def _add_arrow(g, x1, y1, x2, y2, var_from, var_to, css_class='', thin=False):
    """Add an arrow line coloured by source variable."""
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d < 1:
        return
    m1, m2 = _shape_margin(var_from, dx, dy), _shape_margin(var_to, -dx, -dy)
    color = VAR_COLOR.get(var_from, '#888888')
    width = ARROW_W * 0.7 if thin else ARROW_W
    opacity = 0.4 if thin else 0.6
    suffix = '-thin' if thin else ''
    marker = f'url(#ah-{var_from}{suffix})'
    line = dw.Line(
        x1 + dx * m1 / d, y1 + dy * m1 / d,
        x2 - dx * m2 / d, y2 - dy * m2 / d,
        stroke=color, stroke_width=width, stroke_opacity=opacity,
        marker_end=marker,
    )
    if css_class:
        line.args['class'] = css_class
    g.append(line)


def _add_shape(g, cx, cy, label, var, is_output=False, css_class=''):
    """Add a labeled shape (circle / bar / text) to group g."""
    color = VAR_COLOR.get(var, '#2c2c2c')
    opacity = 0.82 if is_output else 0.62
    fw = 700 if is_output else 500
    stype = SHAPE.get(var, 'text')

    sub = dw.Group()
    if css_class:
        sub.args['class'] = css_class

    if stype == 'circle':
        sub.append(dw.Circle(cx, cy, CIRCLE_R,
                             fill=color, fill_opacity=opacity, stroke='none'))
        sub.append(_text(cx, cy, label, FONT, '#ffffff', fw))
    elif stype in ('bar_x', 'bar_y'):
        bw, bh = (BAR if stype == 'bar_x' else BAR[::-1])
        sub.append(dw.Rectangle(cx - bw / 2, cy - bh / 2, bw, bh,
                                rx=BAR_RX, fill=color, fill_opacity=opacity,
                                stroke='none'))
        sub.append(_text(cx, cy, label, FONT_BAR, '#ffffff', fw))
    else:  # text-only (cell center)
        sub.append(_text(cx, cy, label, FONT_CENTER, color, fw))

    g.append(sub)


def _text(x, y, label, size, fill, weight=500):
    return dw.Text(label, size, x, y,
                   text_anchor='middle', dominant_baseline='central',
                   font_family='Arial,sans-serif', font_weight=weight,
                   fill=fill)


# ─── Grid lines ──────────────────────────────────────────────────────────────

def _add_grid(g, cx, cy, out_var, positions, vertical=True, horizontal=True):
    """Add dashed cell-boundary grid lines."""
    xp, yp = GRID_PARITY.get(out_var, (1, 1))
    max_x = max((abs(p[0]) for p in positions), default=1)
    max_y = max((abs(p[1]) for p in positions), default=1)
    margin = 18
    x_lim = max_x * CELL + PAD - 2
    y_lim = max_y * CELL + PAD - 2
    style = dict(stroke=GRID_C, stroke_width=0.9, stroke_dasharray='4,3')

    if vertical:
        for n in range(-6, 7):
            if abs(n) > max_x + 1.5 or n % 2 != xp or abs(n * CELL) > x_lim:
                continue
            g.append(dw.Line(cx + n * CELL, cy - max_y * CELL - margin,
                             cx + n * CELL, cy + max_y * CELL + margin, **style))
    if horizontal:
        for n in range(-6, 7):
            if abs(n) > max_y + 1.5 or n % 2 != yp or abs(n * CELL) > y_lim:
                continue
            g.append(dw.Line(cx - max_x * CELL - margin, cy + n * CELL,
                             cx + max_x * CELL + margin, cy + n * CELL, **style))


# ─── Animation CSS ────────────────────────────────────────────────────────────

NUDGE = 10  # px offset for overlapping shapes

KEYFRAMES = (
    '@keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }\n'
    '@keyframes popIn { 0% { opacity:0; transform:scale(0.7) } '
    '100% { opacity:1; transform:scale(1) } }\n'
    '@keyframes fadeDim { from { opacity: 1 } to { opacity: 0.25 } }\n'
    f'@keyframes nudgeDim {{ from {{ opacity:1; transform:translate(0,0) }} '
    f'to {{ opacity:0.25; transform:translate({NUDGE}px,{NUDGE}px) }} }}\n'
)


def _animation_css(stencil_names):
    """Build CSS rules so each animation phase appears simultaneously.

    Non-composite:  shapes → arrows → output   (3 phases)
    Composite:      Per group: inputs → arrows → intermediates (dim inputs)
                    then: phase-2 arrows → output
                    Total phases = 3 × n_groups + 2
    """
    css = KEYFRAMES
    for name in stencil_names:
        st = STENCILS[name]
        pfx = name

        if st.get('composite'):
            groups = st['phase1_groups']
            n_groups = len(groups)

            # Deduplicate inputs globally, track first/last group usage
            seen = {}  # (x,y) → {idx, first_group, last_group}
            idx = 0
            for g_idx, group in enumerate(groups):
                for inter in group:
                    for inp in inter['inputs']:
                        key = (inp[0], inp[1])
                        if key not in seen:
                            seen[key] = {'idx': idx, 'fg': g_idx, 'lg': g_idx}
                            idx += 1
                        else:
                            seen[key]['lg'] = g_idx

            def t(phase):
                return phase * PHASE_PAUSE

            # Input shapes: appear at first group, dim at last group or output
            output_pos = (0, 0)
            for key, info in seen.items():
                i = info['idx']
                appear = 3 * info['fg']
                if key == output_pos:
                    dim = 3 * n_groups + 1  # nudge when output appears
                    anim = 'nudgeDim'
                else:
                    dim = 3 * info['lg'] + 2  # fade when last consuming group done
                    anim = 'fadeDim'
                css += (f'.s-{pfx}-{i}{{opacity:0;animation:fadeIn .35s ease-out {t(appear):.2f}s both,'
                        f'{anim} .4s ease-out {t(dim):.2f}s forwards}}\n')

            # Thin arrows: one class per group (appear → dim together)
            for g_idx in range(n_groups):
                appear = 3 * g_idx + 1
                dim = 3 * g_idx + 2
                css += (f'.ar-{pfx}-g{g_idx}{{opacity:0;animation:fadeIn .35s ease-out {t(appear):.2f}s both,'
                        f'fadeDim .4s ease-out {t(dim):.2f}s forwards}}\n')

            # Intermediates: one class per group
            for g_idx in range(n_groups):
                appear = 3 * g_idx + 2
                css += (f'.im-{pfx}-g{g_idx}{{opacity:0;animation:popIn .4s ease-out {t(appear):.2f}s both;'
                        f'transform-box:fill-box;transform-origin:center}}\n')

            # Phase-2 arrows
            appear = 3 * n_groups
            css += f'.ar-{pfx}-p2{{opacity:0;animation:fadeIn .35s ease-out {t(appear):.2f}s both}}\n'

            # Output
            appear = 3 * n_groups + 1
            css += (f'.a-{pfx}-out{{opacity:0;animation:popIn .45s ease-out {t(appear):.2f}s both;'
                    f'transform-box:fill-box;transform-origin:center}}\n')
        else:
            n_inp = len(st['inputs'])
            t = [i * PHASE_PAUSE for i in range(3)]

            for i in range(n_inp):
                css += f'.s-{pfx}-{i}{{opacity:0;animation:fadeIn .35s ease-out {t[0]:.2f}s both}}\n'
                css += f'.ar-{pfx}-{i}{{opacity:0;animation:fadeIn .35s ease-out {t[1]:.2f}s both}}\n'
            css += (f'.a-{pfx}-out{{opacity:0;animation:popIn .45s ease-out {t[2]:.2f}s both;'
                    f'transform-box:fill-box;transform-origin:center}}\n')
    return css


# ─── Marker defs ──────────────────────────────────────────────────────────────

def _add_markers(d):
    """Add arrowhead marker definitions for each variable colour."""
    for var, color in VAR_COLOR.items():
        for suffix, mw, mh, rx, ry, fo in [
            ('', 8, 6, 7, 3, 0.6),
            ('-thin', 7, 5, 6, 2.5, 0.4),
        ]:
            mid = f'ah-{var}{suffix}'
            marker = dw.Marker(0, 0, mw, mh, id=mid, orient='auto',
                               markerUnits='strokeWidth')
            marker.args['refX'] = rx
            marker.args['refY'] = ry
            marker.append(dw.Lines(0, 0, mw, ry, 0, mh, mw * 0.25, ry,
                                   fill=color, fill_opacity=fo, close=True))
            d.append_def(marker)


# ─── Stencil rendering ───────────────────────────────────────────────────────

def _all_positions(st):
    """All input/intermediate positions for extent calculation."""
    if st.get('composite'):
        pts = []
        for group in st['phase1_groups']:
            for inter in group:
                pts.append(inter['pos'])
                pts += [(i[0], i[1]) for i in inter['inputs']]
        pts += [(i[0], i[1]) for i in st['phase2_inputs']]
        return pts
    return [(i[0], i[1]) for i in st['inputs']]


def _stencil_dims(name):
    pts = _all_positions(STENCILS[name])
    ex = max(max((abs(p[0]) for p in pts), default=1), 1)
    ey = max(max((abs(p[1]) for p in pts), default=1), 1)
    return 2 * PAD + 2 * ex * CELL, 2 * PAD + 2 * ey * CELL + TITLE_H


def render_stencil(name, parent, ox=0, oy=0, vertical_grid=True, show_title=True):
    """Render one stencil into parent drawing/group at offset (ox, oy)."""
    st = STENCILS[name]
    is_composite = st.get('composite', False)
    pfx = name

    pts = _all_positions(st)
    ex = max(max((abs(p[0]) for p in pts), default=1), 1)
    ey = max(max((abs(p[1]) for p in pts), default=1), 1)
    title_h = TITLE_H if show_title else 0
    w = 2 * PAD + 2 * ex * CELL
    h = 2 * PAD + 2 * ey * CELL + title_h
    cx, cy = PAD + ex * CELL, PAD + ey * CELL + title_h

    g = dw.Group(transform=f'translate({ox},{oy})', id=f'stencil-{pfx}')

    # Grid
    _add_grid(g, cx, cy, st['out_var'], pts, vertical=vertical_grid)

    # Title
    if show_title:
        g.append(_text(w / 2, TITLE_H - 4, st['formula'], 13, TEXT_MUTED, 600))

    if is_composite:
        groups = st['phase1_groups']

        # Deduplicate initial inputs across all groups
        seen, info_map = {}, {}
        idx = 0
        for group in groups:
            for inter in group:
                for inp in inter['inputs']:
                    key = (inp[0], inp[1])
                    if key not in seen:
                        seen[key] = idx
                        info_map[key] = (inp[2], inp[3])
                        idx += 1

        # Thin arrows (behind shapes), one CSS class per group
        for g_idx, group in enumerate(groups):
            for inter in group:
                for inp in inter['inputs']:
                    _add_arrow(g, cx + inp[0] * CELL, cy + inp[1] * CELL,
                               cx + inter['pos'][0] * CELL, cy + inter['pos'][1] * CELL,
                               inp[3], inter['var'],
                               css_class=f'ar-{pfx}-g{g_idx}', thin=True)

        # Initial input shapes
        for key, i in seen.items():
            label, var = info_map[key]
            _add_shape(g, cx + key[0] * CELL, cy + key[1] * CELL,
                       label, var, css_class=f's-{pfx}-{i}')

        # Intermediate shapes, one CSS class per group
        for g_idx, group in enumerate(groups):
            for inter in group:
                tx = cx + inter['pos'][0] * CELL
                ty = cy + inter['pos'][1] * CELL
                _add_shape(g, tx, ty, inter['label'], inter['var'],
                           css_class=f'im-{pfx}-g{g_idx}')

        # Phase-2 arrows
        for i, inp in enumerate(st['phase2_inputs']):
            _add_arrow(g, cx + inp[0] * CELL, cy + inp[1] * CELL, cx, cy,
                       inp[3], st['out_var'], css_class=f'ar-{pfx}-p2')
    else:
        for i, inp in enumerate(st['inputs']):
            ix, iy = cx + inp[0] * CELL, cy + inp[1] * CELL
            _add_arrow(g, ix, iy, cx, cy, inp[3], st['out_var'],
                       css_class=f'ar-{pfx}-{i}')
        for i, inp in enumerate(st['inputs']):
            _add_shape(g, cx + inp[0] * CELL, cy + inp[1] * CELL,
                       inp[2], inp[3], css_class=f's-{pfx}-{i}')

    # Output shape
    _add_shape(g, cx, cy, st['out_label'], st['out_var'],
               is_output=True, css_class=f'a-{pfx}-out')

    parent.append(g)
    return w, h


# ─── Legend ───────────────────────────────────────────────────────────────────

def _add_legend(parent, ox, oy, show_intermediates=False, vertical=False,
                compact=False):
    """Add legend. compact=True uses two-row layout: labels then symbols."""
    items = [('text', 'p', 'cell center'), ('bar_y', 'u', 'x-edge'),
             ('bar_x', 'v', 'y-edge'), ('circle', 'z', 'vertex')]
    if show_intermediates:
        items.append(('text', 'h', 'cell center'))

    g = dw.Group(transform=f'translate({ox},{oy})')
    sc = 0.7

    if compact:
        # Row 1: text labels, left-aligned; Row 2: symbols centered below
        row_sp = 22
        col_sp = 34
        for i, (stype, var, label) in enumerate(items):
            cx = i * col_sp
            g.append(dw.Text(label, 9, cx, 0,
                             text_anchor='middle', dominant_baseline='central',
                             font_family='Arial,sans-serif', fill=TEXT_MUTED))
            color = VAR_COLOR[var]
            sy = row_sp
            if stype == 'circle':
                r = CIRCLE_R * sc
                g.append(dw.Circle(cx, sy, r, fill=color, fill_opacity=0.62,
                                   stroke='none'))
                g.append(_text(cx, sy, var, FONT * sc, '#ffffff', 500))
            elif stype.startswith('bar'):
                bw = BAR[0] * sc if stype == 'bar_x' else BAR[1] * sc
                bh = BAR[1] * sc if stype == 'bar_x' else BAR[0] * sc
                g.append(dw.Rectangle(cx - bw / 2, sy - bh / 2, bw, bh,
                                      rx=BAR_RX * sc, fill=color,
                                      fill_opacity=0.62, stroke='none'))
                g.append(_text(cx, sy, var, FONT_BAR * sc, '#ffffff', 500))
            else:
                g.append(_text(cx, sy, var, FONT_CENTER * sc, color, 600))
    else:
        x, y = 0, 0
        for stype, var, label in items:
            desc = f'{var} — {label}'
            color = VAR_COLOR[var]
            if stype == 'circle':
                r = CIRCLE_R * sc
                g.append(dw.Circle(x + r, y, r, fill=color, fill_opacity=0.62,
                                   stroke='none'))
                g.append(_text(x + 2 * r + 6 + 40, y, desc, 11, TEXT_MUTED))
            elif stype.startswith('bar'):
                bw = BAR[0] * sc if stype == 'bar_x' else BAR[1] * sc
                bh = BAR[1] * sc if stype == 'bar_x' else BAR[0] * sc
                g.append(dw.Rectangle(x, y - bh / 2, bw, bh, rx=BAR_RX * sc,
                                      fill=color, fill_opacity=0.62,
                                      stroke='none'))
                g.append(_text(x + bw + 6 + 40, y, desc, 11, TEXT_MUTED))
            else:
                g.append(_text(x, y, var, 12, color, 600))
                g.append(_text(x + 14 + 40, y, f'— {label}', 11, TEXT_MUTED))
            if vertical:
                y += 28
            else:
                x += 140
    parent.append(g)


# ─── Combined SVG builders ───────────────────────────────────────────────────

def _make_drawing(width, height, css, stencil_names):
    d = dw.Drawing(width, height)
    d.append(dw.Raw(f'<style>\n{css}</style>'))
    _add_markers(d)
    return d


def generate_phase1():
    names = ['cu', 'cv', 'z', 'h']
    sub_w, sub_h = _stencil_dims(names[0])
    cols, rows = 2, 2
    total_w = cols * sub_w + (cols - 1) * GAP
    total_h = rows * sub_h + (rows - 1) * GAP + 40

    d = _make_drawing(total_w, total_h, _animation_css(names), names)
    for idx, name in enumerate(names):
        render_stencil(name, d, (idx % cols) * (sub_w + GAP),
                       (idx // cols) * (sub_h + GAP))
    _add_legend(d, 20, total_h - 20)
    return d


def generate_phase2():
    names = ['unew', 'vnew', 'pnew']
    sub_w, sub_h = _stencil_dims(names[0])
    total_w = 3 * sub_w + 2 * GAP
    total_h = sub_h + 40

    d = _make_drawing(total_w, total_h, _animation_css(names), names)
    for i, name in enumerate(names):
        render_stencil(name, d, i * (sub_w + GAP), 0)
    _add_legend(d, 20, total_h - 20, show_intermediates=True)
    return d


def generate_composite():
    names = ['p_composite', 'v_composite', 'u_composite']

    # Tight two-row layout: shared grid cells overlap exactly (no titles).
    # Each sub-stencil center at (PAD + 2·CELL, PAD + 2·CELL) = (182, 182).
    # sub_dim (no title) = 2·PAD + 4·CELL = 364.
    #
    # p at (0,0):  rightmost p at (2,0) → pixel (312, 182)
    # v at (4·CELL, CELL): p at (-2,-1) → pixel (260+52, 65+117) = (312, 182) ✓
    # u at (CELL, 4·CELL): p at (-1,-2) → pixel (65+117, 260+52) = (182, 312) ✓
    #   which matches p's p at (0,2) → (182, 312) ✓
    #
    # Grid alignment: p↔v Δx=4·CELL(even,same xp=1✓) Δy=CELL(odd,yp differs✓)
    #                 p↔u Δx=CELL(odd,xp differs✓) Δy=4·CELL(even,same yp=1✓)
    offsets = {
        'p_composite': (0, 0),
        'v_composite': (4 * CELL, CELL),
        'u_composite': (CELL, 4 * CELL),
    }

    sub_dim = 2 * PAD + 4 * CELL  # 364 (no title)
    total_w = max(ox + sub_dim for ox, _ in offsets.values())
    total_h = max(oy + sub_dim for _, oy in offsets.values())

    d = _make_drawing(total_w, total_h, _animation_css(names), names)
    for name in names:
        ox, oy = offsets[name]
        render_stencil(name, d, ox, oy, show_title=False)
    # Compact legend in the bottom-right empty area
    _add_legend(d, total_w - 5 * 34 + 10, total_h - 52,
                show_intermediates=True, compact=True)
    return d


def generate_individual():
    results = {}
    for name in STENCILS:
        d = _make_drawing(*_stencil_dims(name), _animation_css([name]), [name])
        render_stencil(name, d, 0, 0)
        results[name] = d
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global PHASE_PAUSE

    parser = argparse.ArgumentParser(description='Generate SWM stencil SVGs')
    parser.add_argument('--phase-pause', type=float, default=PHASE_PAUSE,
                        help=f'Pause between animation phases (default: {PHASE_PAUSE})')
    args = parser.parse_args()
    PHASE_PAUSE = args.phase_pause

    out = os.path.normpath(OUTPUT_DIR)
    os.makedirs(out, exist_ok=True)

    for label, gen, fname in [
        ('intermediates', generate_phase1, 'swm_intermediates.svg'),
        ('updates',       generate_phase2, 'swm_updates.svg'),
        ('composite',     generate_composite, 'swm_composite.svg'),
    ]:
        path = os.path.join(out, fname)
        gen().save_svg(path)
        print(f'  wrote {path}')

    for name, d in generate_individual().items():
        path = os.path.join(out, f'swm_{name}.svg')
        d.save_svg(path)
        print(f'  wrote {path}')

    print(f'\nDone — SVGs written to {out}/')


if __name__ == '__main__':
    main()
