#!/usr/bin/env python3
"""
Generate SVG stencil diagrams for the Shallow Water Model (SWM) operators.

Illustrates the Arakawa C-grid staggering and the stencil patterns for:
  Phase 1 — intermediate quantities: cu, cv, z, h
  Phase 2 — time-step updates:       ũ, ṽ, p̃

Shapes encode grid location:
  ● circle    → cell center  (p, h)
  ◆ diamond_x → x-edge       (u, cu)  — wider horizontally
  ◆ diamond_y → y-edge       (v, cv)  — taller vertically
  ■ square    → vertex        (z)

Output SVGs use the CSCS Reveal.js color palette.
"""

import math
import os

# ─── Configuration ────────────────────────────────────────────────────────────

CELL = 65          # px between adjacent staggered positions
R = 17             # base shape radius
R_LONG = 21        # diamond long axis
R_SHORT = 13       # diamond short axis
SMALL_R = 11       # same-position input shape radius
SMALL_RL = 14
SMALL_RS = 9
FONT = 12
FONT_SM = 10
STROKE = 1.6
STROKE_OUT = 2.2
ARROW_W = 1.1
PAD = 52           # padding around each sub-diagram
TITLE_H = 24       # space for title above diagram
ANIM_DT = 0.13     # seconds between animation steps
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
GRID_C = '#e4e6ea'
TEXT_C = '#2c2c2c'
TEXT_MUTED = '#888888'

# Shape type per variable
SHAPE = {
    'p': 'circle',
    'u': 'diamond_x',
    'v': 'diamond_y',
    'z': 'square',
    'h': 'circle',
}


# ─── Stencil definitions ─────────────────────────────────────────────────────
# Each input: (dx, dy, label, var_type)
# dx, dy in CELL units relative to output at (0, 0)

STENCILS = {
    # Phase 1: intermediate quantities
    'cu': {
        'formula': 'cu = ⟨p⟩ₓ · u',
        'out_label': 'cu', 'out_var': 'u',
        'same': ('u', 'u'),
        'inputs': [
            (-1, 0, 'p', 'p'),
            ( 1, 0, 'p', 'p'),
        ],
    },
    'cv': {
        'formula': 'cv = ⟨p⟩ᵧ · v',
        'out_label': 'cv', 'out_var': 'v',
        'same': ('v', 'v'),
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
        'same': ('p', 'p'),
        'inputs': [
            (-1, 0, 'u', 'u'), (1, 0, 'u', 'u'),
            (0, -1, 'v', 'v'), (0, 1, 'v', 'v'),
        ],
    },
    # Phase 2: time-step updates
    'unew': {
        'formula': 'ũ (u update)',
        'out_label': 'ũ', 'out_var': 'u',
        'same': ('U', 'u'),
        'inputs': [
            ( 0, -1, 'z', 'z'), ( 0,  1, 'z', 'z'),
            (-1,  0, 'h', 'h'), ( 1,  0, 'h', 'h'),
            (-1, -1, 'V', 'v'), ( 1, -1, 'V', 'v'),
            (-1,  1, 'V', 'v'), ( 1,  1, 'V', 'v'),
        ],
    },
    'vnew': {
        'formula': 'ṽ (v update)',
        'out_label': 'ṽ', 'out_var': 'v',
        'same': ('V', 'v'),
        'inputs': [
            (-1,  0, 'z', 'z'), ( 1,  0, 'z', 'z'),
            ( 0, -1, 'h', 'h'), ( 0,  1, 'h', 'h'),
            (-1, -1, 'U', 'u'), ( 1, -1, 'U', 'u'),
            (-1,  1, 'U', 'u'), ( 1,  1, 'U', 'u'),
        ],
    },
    'pnew': {
        'formula': 'p\u0303 (p update)',
        'out_label': 'p\u0303', 'out_var': 'p',
        'same': ('p', 'p'),
        'inputs': [
            (-1, 0, 'U', 'u'), (1, 0, 'U', 'u'),
            (0, -1, 'V', 'v'), (0, 1, 'V', 'v'),
        ],
    },
}


# ─── SVG element helpers ─────────────────────────────────────────────────────

def _shape_path(cx, cy, stype, r=R, rl=R_LONG, rs=R_SHORT):
    """Return SVG element (without closing) for a shape."""
    if stype == 'circle':
        return f'<circle cx="{cx}" cy="{cy}" r="{r}"'
    elif stype == 'diamond_x':
        pts = f'{cx - rl},{cy} {cx},{cy - rs} {cx + rl},{cy} {cx},{cy + rs}'
        return f'<polygon points="{pts}"'
    elif stype == 'diamond_y':
        pts = f'{cx - rs},{cy} {cx},{cy - rl} {cx + rs},{cy} {cx},{cy + rl}'
        return f'<polygon points="{pts}"'
    elif stype == 'square':
        s = r * 0.82
        return f'<rect x="{cx - s}" y="{cy - s}" width="{2 * s}" height="{2 * s}"'
    return ''


def shape_svg(cx, cy, label, var, is_output=False, small=False, cls=''):
    """Generate SVG group for a shape with label."""
    color = VC.get(var, TEXT_C)
    stype = SHAPE.get(var, 'circle')
    if small:
        r, rl, rs, fs, sw = SMALL_R, SMALL_RL, SMALL_RS, FONT_SM, STROKE
    elif is_output:
        r, rl, rs, fs, sw = R, R_LONG, R_SHORT, FONT, STROKE_OUT
    else:
        r, rl, rs, fs, sw = R, R_LONG, R_SHORT, FONT, STROKE

    fw = '700' if is_output else '500'
    cls_attr = f' class="{cls}"' if cls else ''

    s = f'<g{cls_attr}>\n'
    if is_output:
        s += f'  {_shape_path(cx, cy, stype, r, rl, rs)} fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="{sw}"/>\n'
    else:
        s += f'  {_shape_path(cx, cy, stype, r, rl, rs)} fill="#ffffff" stroke="{color}" stroke-width="{sw}"/>\n'
    s += (f'  <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
          f'font-family="Arial,sans-serif" font-size="{fs}" font-weight="{fw}" '
          f'fill="{color}">{label}</text>\n')
    s += '</g>\n'
    return s


def arrow_svg(x1, y1, x2, y2, cls=''):
    """Generate SVG arrow line from (x1,y1) toward (x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d < 1:
        return ''
    margin = R + 5
    f = margin / d
    ax1 = x1 + dx * f
    ay1 = y1 + dy * f
    ax2 = x2 - dx * f
    ay2 = y2 - dy * f
    cls_attr = f' class="{cls}"' if cls else ''
    return (f'<line{cls_attr} x1="{ax1:.1f}" y1="{ay1:.1f}" '
            f'x2="{ax2:.1f}" y2="{ay2:.1f}" '
            f'stroke="{ARROW_C}" stroke-width="{ARROW_W}" marker-end="url(#ah)"/>\n')


def grid_lines_svg(cx, cy, inputs):
    """Draw faint background grid lines based on actual stencil extent."""
    # Determine which grid lines are needed from input positions
    xs = sorted(set([0] + [inp[0] for inp in inputs]))
    ys = sorted(set([0] + [inp[1] for inp in inputs]))
    x_ext = max(abs(xs[0]), abs(xs[-1])) if xs else 1
    y_ext = max(abs(ys[0]), abs(ys[-1])) if ys else 1
    margin = 15
    lines = ''
    for nx in xs:
        x = cx + nx * CELL
        lines += (f'<line x1="{x}" y1="{cy - y_ext * CELL - margin}" '
                  f'x2="{x}" y2="{cy + y_ext * CELL + margin}" '
                  f'stroke="{GRID_C}" stroke-width="0.8" stroke-dasharray="3,4"/>\n')
    for ny in ys:
        y = cy + ny * CELL
        lines += (f'<line x1="{cx - x_ext * CELL - margin}" y1="{y}" '
                  f'x2="{cx + x_ext * CELL + margin}" y2="{y}" '
                  f'stroke="{GRID_C}" stroke-width="0.8" stroke-dasharray="3,4"/>\n')
    return lines


# ─── Stencil diagram generation ──────────────────────────────────────────────

def render_stencil(name, ox=0, oy=0, animate=True, id_prefix=''):
    """Render a stencil diagram as an SVG <g> group.

    Returns (svg_string, width, height, num_anim_steps).
    """
    st = STENCILS[name]
    inputs = st['inputs']
    same = st.get('same')

    # Sub-diagram dimensions
    w = 2 * PAD + 2 * CELL
    h = 2 * PAD + 2 * CELL + TITLE_H

    # Center of the stencil pattern
    cx = PAD + CELL
    cy = PAD + CELL + TITLE_H

    pfx = id_prefix or name
    svg = f'<g transform="translate({ox},{oy})" id="stencil-{pfx}">\n'

    # Background grid
    svg += grid_lines_svg(cx, cy, inputs)

    # Title
    svg += (f'<text x="{w / 2}" y="{TITLE_H - 4}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="13" font-weight="600" '
            f'fill="{TEXT_MUTED}">{st["formula"]}</text>\n')

    step = 0
    # Arrows first (drawn behind shapes)
    for inp in inputs:
        ix = cx + inp[0] * CELL
        iy = cy + inp[1] * CELL
        cls = f'a-{pfx}-{step}' if animate else ''
        svg += arrow_svg(ix, iy, cx, cy, cls)
        step += 1

    # Input shapes (on top of arrows)
    step = 0
    for inp in inputs:
        ix = cx + inp[0] * CELL
        iy = cy + inp[1] * CELL
        cls = f'a-{pfx}-{step}' if animate else ''
        svg += shape_svg(ix, iy, inp[2], inp[3], cls=cls)
        step += 1

    # Same-position input (small, offset)
    if same:
        cls = f'a-{pfx}-{step}' if animate else ''
        svg += shape_svg(cx + 15, cy + 14, same[0], same[1], small=True, cls=cls)
        step += 1

    # Output shape (drawn last, on top)
    cls = f'a-{pfx}-out' if animate else ''
    svg += shape_svg(cx, cy, st['out_label'], st['out_var'], is_output=True, cls=cls)
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
        n_inputs = len(st['inputs'])
        has_same = st.get('same') is not None

        for i in range(n_inputs):
            delay = i * ANIM_DT
            css += f'.a-{pfx}-{i} {{ opacity:0; animation: fadeIn 0.25s ease-out {delay:.2f}s both; }}\n'

        if has_same:
            delay = n_inputs * ANIM_DT
            css += f'.a-{pfx}-{n_inputs} {{ opacity:0; animation: fadeIn 0.25s ease-out {delay:.2f}s both; }}\n'

        out_delay = (n_inputs + (1 if has_same else 0)) * ANIM_DT + 0.1
        css += (f'.a-{pfx}-out {{ opacity:0; animation: popIn 0.35s ease-out {out_delay:.2f}s both; '
                f'transform-box: fill-box; transform-origin: center; }}\n')

    return css


def svg_defs():
    """Return common SVG <defs> (arrowhead marker)."""
    return '''<defs>
  <marker id="ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L8,3 L0,6 L2,3 Z" fill="#b5b5b5"/>
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
        ('circle', 'p', 'p — cell center'),
        ('diamond_x', 'u', 'u — x-edge'),
        ('diamond_y', 'v', 'v — y-edge'),
        ('square', 'z', 'z — vertex'),
    ]
    if show_intermediates:
        items.append(('circle', 'h', 'h — Bernoulli'))

    svg = f'<g transform="translate({ox},{oy})">\n'
    x = 0
    for stype, var, desc in items:
        color = VC[var]
        r, rl, rs = SMALL_R, SMALL_RL, SMALL_RS
        svg += f'  {_shape_path(x + SMALL_R, 0, stype, r, rl, rs)} fill="#ffffff" stroke="{color}" stroke-width="{STROKE}"/>\n'
        svg += (f'  <text x="{x + SMALL_R * 2 + 6}" y="0" dominant-baseline="central" '
                f'font-family="Arial,sans-serif" font-size="11" fill="{TEXT_MUTED}">{desc}</text>\n')
        x += 140
    svg += '</g>\n'
    return svg


# ─── Combined SVG generators ─────────────────────────────────────────────────

def generate_phase1_svg():
    """Generate combined SVG for Phase 1 intermediates (cu, cv, z, h)."""
    names = ['cu', 'cv', 'z', 'h']
    # 2x2 grid layout
    sub_w = 2 * PAD + 2 * CELL
    sub_h = 2 * PAD + 2 * CELL + TITLE_H

    cols, rows = 2, 2
    total_w = cols * sub_w + (cols - 1) * GAP
    total_h = rows * sub_h + (rows - 1) * GAP + 40  # +40 for legend

    content = ''
    for idx, name in enumerate(names):
        col = idx % cols
        row = idx // cols
        ox = col * (sub_w + GAP)
        oy = row * (sub_h + GAP)
        svg_part, _, _, _ = render_stencil(name, ox, oy)
        content += svg_part

    # Legend at bottom
    content += legend_svg(20, total_h - 20)

    css = animation_css(names)
    return wrap_svg(content, total_w, total_h, css)


def generate_phase2_svg():
    """Generate combined SVG for Phase 2 updates (ũ, ṽ, p̃)."""
    names = ['unew', 'vnew', 'pnew']
    sub_w = 2 * PAD + 2 * CELL
    sub_h = 2 * PAD + 2 * CELL + TITLE_H

    cols = 3
    total_w = cols * sub_w + (cols - 1) * GAP
    total_h = sub_h + 40  # +40 for legend

    content = ''
    for idx, name in enumerate(names):
        ox = idx * (sub_w + GAP)
        svg_part, _, _, _ = render_stencil(name, ox, 0)
        content += svg_part

    content += legend_svg(20, total_h - 20, show_intermediates=True)

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
    out_dir = os.path.normpath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # Combined SVGs
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

    # Individual SVGs
    individuals = generate_individual_svgs()
    for name, svg in individuals.items():
        path = os.path.join(out_dir, f'swm_{name}.svg')
        with open(path, 'w') as f:
            f.write(svg)
        print(f'  wrote {path}')

    print(f'\nDone — {2 + len(individuals)} SVGs written to {out_dir}/')


if __name__ == '__main__':
    main()
