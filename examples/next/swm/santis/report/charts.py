"""Minimal SVG chart helpers for the report: log/linear axes, IQR error bars, direct labels."""

import math

W, H = 760, 400
ML, MR, MT, MB = 64, 24, 20, 48


def _fmt(v):
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:.0f}" if abs(v) < 1e5 else f"{v:.0e}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if abs(v) >= 0.01:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _ticks_log(lo, hi):
    ticks = []
    e = math.floor(math.log10(lo))
    while 10**e <= hi * 1.0001:
        for m in (1, 2, 5):
            v = m * 10**e
            if lo * 0.999 <= v <= hi * 1.001:
                ticks.append(v)
        e += 1
    if len(ticks) > 9:
        ticks = [t for t in ticks if abs(math.log10(t) - round(math.log10(t))) < 1e-9]
    return ticks


def _ticks_lin(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n + 1:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


class Chart:
    def __init__(
        self,
        title,
        xlabel,
        ylabel,
        xlog=True,
        ylog=True,
        w=W,
        h=H,
        xticks=None,
        xtickfmt=None,
        ylim=None,
    ):
        self.title, self.xlabel, self.ylabel, self.xlog, self.ylog = (
            title,
            xlabel,
            ylabel,
            xlog,
            ylog,
        )
        self.w, self.h = w, h
        self.series = []
        self.hlines = []
        self.xticks = xticks
        self.xtickfmt = xtickfmt or _fmt
        self.ylim = ylim
        self.marks = []

    def add(self, name, pts, color, dash=None, marker="circle", label=True, width=2):
        """pts: list of (x, y, lo, hi) or (x, y)."""
        self.series.append(
            dict(
                name=name,
                pts=[p if len(p) == 4 else (p[0], p[1], None, None) for p in pts],
                color=color,
                dash=dash,
                marker=marker,
                label=label,
                width=width,
            )
        )

    def hline(self, y, text, color="var(--ink-3)"):
        self.hlines.append((y, text, color))

    def mark(self, x, y, text, color):
        self.marks.append((x, y, text, color))

    def _scales(self):
        xs = [p[0] for s in self.series for p in s["pts"]] + [m[0] for m in self.marks]
        ys = (
            [v for s in self.series for p in s["pts"] for v in (p[1], p[2], p[3]) if v is not None]
            + [h[0] for h in self.hlines]
            + [m[1] for m in self.marks]
        )
        xlo, xhi = min(xs), max(xs)
        ylo, yhi = min(ys), max(ys)
        if self.ylim:
            ylo, yhi = self.ylim
        if self.xlog:
            xlo, xhi = xlo / 1.3, xhi * 1.3
        else:
            pad = (xhi - xlo) * 0.05 or 1
            xlo, xhi = xlo - pad, xhi + pad
        if self.ylog:
            ylo, yhi = ylo / 1.4, yhi * 1.4
        else:
            if not self.ylim:
                ylo = min(0, ylo)
                yhi = yhi * 1.12
        mr = 150 if any(s["label"] for s in self.series) else MR
        pw, ph = self.w - ML - mr, self.h - MT - MB
        fx = (
            (
                lambda x: ML
                + (math.log10(x) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) * pw
            )
            if self.xlog
            else (lambda x: ML + (x - xlo) / (xhi - xlo) * pw)
        )
        fy = (
            (
                lambda y: MT
                + ph
                - (math.log10(y) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) * ph
            )
            if self.ylog
            else (lambda y: MT + ph - (y - ylo) / (yhi - ylo) * ph)
        )
        return xlo, xhi, ylo, yhi, fx, fy, pw, ph

    def svg(self):
        xlo, xhi, ylo, yhi, fx, fy, pw, ph = self._scales()
        o = [
            f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{self.title}" class="chart">'
        ]
        o.append(f'<text x="{ML}" y="{MT - 6}" class="ct">{self.title}</text>')
        xt = self.xticks or (_ticks_log(xlo, xhi) if self.xlog else _ticks_lin(xlo, xhi))
        yt = _ticks_log(ylo, yhi) if self.ylog else _ticks_lin(ylo, yhi)
        for t in yt:
            y = fy(t)
            o.append(
                f'<line x1="{ML}" x2="{ML + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{ML - 8}" y="{y + 4:.1f}" class="tk" text-anchor="end">{_fmt(t)}</text>'
            )
        for t in xt:
            if not (xlo <= t <= xhi):
                continue
            x = fx(t)
            o.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT + ph}" y2="{MT + ph + 4}" class="ax"/><text x="{x:.1f}" y="{MT + ph + 18}" class="tk" text-anchor="middle">{self.xtickfmt(t)}</text>'
            )
        o.append(f'<line x1="{ML}" x2="{ML + pw}" y1="{MT + ph}" y2="{MT + ph}" class="ax"/>')
        o.append(
            f'<text x="{ML + pw / 2}" y="{self.h - 8}" class="al" text-anchor="middle">{self.xlabel}</text>'
        )
        o.append(
            f'<text transform="translate(14,{MT + ph / 2}) rotate(-90)" class="al" text-anchor="middle">{self.ylabel}</text>'
        )
        for y, text, color in self.hlines:
            yy = fy(y)
            o.append(
                f'<line x1="{ML}" x2="{ML + pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-dasharray="3 4" stroke-width="1.5"/><text x="{ML + pw - 4}" y="{yy - 5:.1f}" class="tk" text-anchor="end" fill="{color}">{text}</text>'
            )
        labels = []
        for s in self.series:
            pts = [p for p in s["pts"] if p[1] is not None]
            if not pts:
                continue
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{fx(p[0]):.1f},{fy(p[1]):.1f}" for i, p in enumerate(pts)
            )
            dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
            if s["width"]:
                o.append(
                    f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{s["width"]}"{dash} stroke-linejoin="round"/>'
                )
            for p in pts:
                x, y = fx(p[0]), fy(p[1])
                if p[2] is not None and p[3] is not None and p[3] > p[2]:
                    o.append(
                        f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{fy(p[3]):.1f}" y2="{fy(p[2]):.1f}" stroke="{s["color"]}" stroke-width="1.5"/><line x1="{x - 3:.1f}" x2="{x + 3:.1f}" y1="{fy(p[3]):.1f}" y2="{fy(p[3]):.1f}" stroke="{s["color"]}" stroke-width="1.5"/><line x1="{x - 3:.1f}" x2="{x + 3:.1f}" y1="{fy(p[2]):.1f}" y2="{fy(p[2]):.1f}" stroke="{s["color"]}" stroke-width="1.5"/>'
                    )
                if s["marker"] == "none":
                    continue
                big = s["name"].endswith("(whole step)")
                r_ = 7 if big else 4
                shape = (
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_}" fill="{s["color"]}" stroke="var(--paper)" stroke-width="1.5"/>'
                    if s["marker"] == "circle"
                    else f'<rect x="{x - r_:.1f}" y="{y - r_:.1f}" width="{2 * r_}" height="{2 * r_}" fill="{s["color"]}" stroke="var(--paper)" stroke-width="1.5"/>'
                )
                tip = f"{s['name']}: x={self.xtickfmt(p[0])}, {_fmt(p[1])}" + (
                    f" [IQR {_fmt(p[2])}–{_fmt(p[3])}]" if p[2] is not None else ""
                )
                o.append(f'<g class="pt">{shape}<title>{tip}</title></g>')
            if s["label"]:
                labels.append([fx(pts[-1][0]) + 7, fy(pts[-1][1]) + 4, s["name"], s["color"]])
        labels.sort(key=lambda l: l[1])
        for i in range(1, len(labels)):
            if labels[i][1] - labels[i - 1][1] < 13:
                labels[i][1] = labels[i - 1][1] + 13
        for lx, ly, name, color in labels:
            o.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="dl" fill="{color}">{name}</text>')
        for x, y, text, color in self.marks:
            o.append(
                f'<text x="{fx(x):.1f}" y="{fy(y) - 8:.1f}" class="dl" fill="{color}" text-anchor="middle">{text}</text>'
            )
        o.append("</svg>")
        return "\n".join(o)


def bars(title, groups, series, color_of, ylabel, w=W, h=340, ylog=False, fmt=_fmt):
    """groups: list of group labels; series: {name: [values per group]} (None = missing)."""
    pw, ph = w - ML - MR, h - MT - MB
    vals = [v for s in series.values() for v in s if v is not None]
    ylo, yhi = (min(vals) / 1.5, max(vals) * 1.5) if ylog else (0, max(vals) * 1.15)
    fy = (
        (
            lambda y: MT
            + ph
            - (math.log10(y) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) * ph
        )
        if ylog
        else (lambda y: MT + ph - (y - ylo) / (yhi - ylo) * ph)
    )
    n, k = len(groups), len(series)
    gw = pw / n
    bw = gw * 0.8 / k
    o = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}" class="chart">',
        f'<text x="{ML}" y="{MT - 6}" class="ct">{title}</text>',
    ]
    yt = _ticks_log(ylo, yhi) if ylog else _ticks_lin(ylo, yhi)
    for t in yt:
        y = fy(t)
        o.append(
            f'<line x1="{ML}" x2="{ML + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{ML - 8}" y="{y + 4:.1f}" class="tk" text-anchor="end">{_fmt(t)}</text>'
        )
    for gi, g in enumerate(groups):
        x0 = ML + gi * gw + gw * 0.1
        for si, (name, s) in enumerate(series.items()):
            v = s[gi]
            if v is None:
                continue
            x = x0 + si * bw
            y = fy(v)
            base = fy(ylo) if ylog else fy(0)
            o.append(
                f'<g class="pt"><rect x="{x + 1:.1f}" y="{min(y, base):.1f}" width="{bw - 2:.1f}" height="{abs(base - y):.1f}" fill="{color_of(name)}" rx="2"/><title>{name}, {g}: {fmt(v)}</title></g>'
            )
        o.append(
            f'<text x="{x0 + gw * 0.4:.1f}" y="{MT + ph + 18}" class="tk" text-anchor="middle">{g}</text>'
        )
    o.append(f'<line x1="{ML}" x2="{ML + pw}" y1="{MT + ph}" y2="{MT + ph}" class="ax"/>')
    o.append(
        f'<text transform="translate(14,{MT + ph / 2}) rotate(-90)" class="al" text-anchor="middle">{ylabel}</text>'
    )
    o.append("</svg>")
    return "\n".join(o)


def legend(items):
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i style="background:{c}{";border-style:dashed" if d else ""}"></i>{n}</span>'
            for n, c, d in items
        )
        + "</div>"
    )
