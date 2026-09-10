"""Assemble the Santis GPU scaling report (HTML with inline SVG charts) from report_data.json."""

import json, math, html
import charts as C

D = json.load(open("report_data.json"))
P = D["P"]
SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
COL = {
    "padded": "var(--s1)",
    "coloured2ph": "var(--s2)",
    "coloured8": "var(--s3)",
    "allgather": "var(--s4)",
    "ragged": "var(--s5)",
    "ref": "var(--ink-3)",
}
PCOL = {
    1: "var(--o1)",
    2: "var(--o2)",
    4: "var(--o3)",
    8: "var(--o4)",
    16: "var(--o5)",
    32: "var(--o6)",
}
TNAME = {
    "padded": "padded all-to-all",
    "coloured2ph": "coloured ppermute, 2 phases",
    "coloured8": "coloured ppermute, 8 rounds",
    "allgather": "all-gather",
    "ragged": "ragged all-to-all",
}


def ser(d, keys=None):
    keys = keys or sorted(int(k) for k in d)
    return [(k, d[str(k)]["med"], d[str(k)]["q25"], d[str(k)]["q75"]) for k in keys if str(k) in d]


def g(mode, t, run):  # strong series dict keyed by size (str)
    return D["strong"][mode][t].get(run, {})


def med(mode, t, run, size):
    r = g(mode, t, run).get(str(size))
    return r["med"] if r else None


def fmt(v, nd=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, str):
        return v
    if nd is not None:
        return f"{v:.{nd}f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def pct(v):
    return "–" if v is None else f"{100 * v:.0f}%"


def table(head, rows, cls="data"):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fig(svg, caption, legend=""):
    return f"<figure>{svg}{legend}<figcaption>{caption}</figcaption></figure>"


# ---------- derived numbers used in prose ----------
ref = D["ref"]
base16 = med("fwd", "padded", "single:1x1", 16384)
ref16 = ref["ref"]["16384"]["med"]
p32 = med("fwd", "padded", "multi:32x1", 16384)
c32 = med("fwd", "coloured2ph", "multi:32x1", 16384)
c8_32 = med("fwd", "coloured8", "multi:32x1", 16384)
sp_p32, sp_c32, sp_c8 = (
    base16 / p32,
    med("fwd", "coloured2ph", "single:1x1", 16384) / c32,
    med("fwd", "coloured8", "single:1x1", 16384) / c8_32,
)
gr1 = med("grad_remat", "padded", "single:1x1", 8192)
gr32 = med("grad_remat", "padded", "multi:32x1", 8192)
gr16 = med("grad_remat", "padded", "multi:16x1", 8192)
grc32 = med("grad_remat", "coloured2ph", "multi:32x1", 8192)
grc1 = med("grad_remat", "coloured2ph", "single:1x1", 8192)
overhead16 = base16 / ref16
ad_ref = ref["ref_grad"]["2048"]["med"] / ref["ref"]["2048"]["med"]
ad_ref_remat = ref["ref_grad_remat"]["4096"]["med"] / ref["ref"]["4096"]["med"]
remat_vs_grad_2048 = med("grad_remat", "padded", "single:1x1", 2048) / med(
    "grad", "padded", "single:1x1", 2048
)
mem = D["mem_gb"]
env = D["env"]


# ---------- charts ----------
def strong_chart(mode, t, title, ylabel="ms per step (median, IQR)"):
    c = C.Chart(title, "global grid M = N", ylabel, xticks=SIZES, xtickfmt=lambda v: f"{int(v)}")
    if mode == "fwd":
        c.add(
            "reference, 1 GPU", ser(ref["ref"]), COL["ref"], dash="5 4", label=True, marker="square"
        )
    if mode == "grad_remat":
        c.add(
            "reference grad (remat), 1 GPU",
            ser(ref["ref_grad_remat"]),
            COL["ref"],
            dash="5 4",
            label=True,
            marker="square",
        )
    for run, lab in [
        ("single:1x1", "P=1"),
        ("multi:2x1", "P=2"),
        ("multi:2x2", "P=4 (2×2)"),
        ("multi:8x1", "P=8 (8×1)"),
        ("multi:16x1", "P=16 (16×1)"),
        ("multi:32x1", "P=32 (32×1)"),
    ]:
        d = g(mode, t, run)
        if d:
            c.add(lab, ser(d), PCOL[P[run]], label=True)
    return c.svg()


def eff_chart(mode, size, title):
    c = C.Chart(
        title,
        "GPUs (4 per node above P=4)",
        "parallel efficiency vs 1 GPU",
        xlog=True,
        ylog=False,
        xticks=[1, 2, 4, 8, 16, 32],
        xtickfmt=lambda v: f"{int(v)}",
        ylim=(0, 1.15),
    )
    for t in ["padded", "coloured2ph", "coloured8", "allgather", "ragged"]:
        if mode == "grad_remat" and t == "ragged":
            continue
        b = med(mode, t, "single:1x1", size)
        if not b:
            continue
        pts = []
        for run in [
            "single:1x1",
            "multi:2x1",
            "multi:2x2",
            "multi:8x1",
            "multi:16x1",
            "multi:32x1",
        ]:
            r = g(mode, t, run).get(str(size))
            if r:
                pts.append(
                    (P[run], b / r["med"] / P[run], b / r["q75"] / P[run], b / r["q25"] / P[run])
                )
        c.add(TNAME[t], pts, COL[t], label=False)
    c.hline(1.0, "ideal")
    return c.svg()


def weak_chart(mode, title):
    c = C.Chart(
        title,
        "GPUs (4 per node above P=4)",
        "weak-scaling efficiency vs 1 GPU",
        xlog=True,
        ylog=False,
        xticks=[1, 2, 4, 8, 16, 32],
        xtickfmt=lambda v: f"{int(v)}",
        ylim=(0, 1.15),
    )
    W = D["weak"][mode]["padded"]
    for i, edge in enumerate([256, 1024, 2048, 4096]):
        b = W["single:1x1"].get(str(edge))
        if not b:
            continue
        pts = []
        for run in [
            "single:1x1",
            "multi:2x1",
            "multi:2x2",
            "multi:8x1",
            "multi:16x1",
            "multi:32x1",
        ]:
            r = W.get(run, {}).get(str(edge))
            if r:
                pts.append((P[run], b["med"] / r["med"], b["med"] / r["q75"], b["med"] / r["q25"]))
        c.add(
            f"{edge}² per GPU",
            pts,
            ["var(--w1)", "var(--w2)", "var(--w3)", "var(--w4)"][i],
            label=False,
        )
    c.hline(1.0, "ideal")
    return c.svg()


def transports_bar():
    groups = ["P=1", "P=4 (2×2)", "P=8 (8×1)", "P=16 (16×1)", "P=32 (32×1)"]
    runs = ["single:1x1", "multi:2x2", "multi:8x1", "multi:16x1", "multi:32x1"]
    series = {
        TNAME[t]: [med("fwd", t, r, 8192) for r in runs]
        for t in ["padded", "coloured2ph", "coloured8", "allgather", "ragged"]
    }
    return C.bars(
        "Forward step at 8192², all transports",
        groups,
        series,
        lambda n: COL[[k for k, v in TNAME.items() if v == n][0]],
        "ms per step (median)",
        ylog=True,
    )


def ad_cost_chart():
    c = C.Chart(
        "Cost of the gradient relative to the forward step",
        "global grid M = N",
        "grad time / forward time",
        xticks=SIZES,
        xtickfmt=lambda v: f"{int(v)}",
        ylog=False,
        ylim=(0, 5.5),
    )
    rr = [
        (int(k), ref["ref_grad"][k]["med"] / ref["ref"][k]["med"])
        for k in ref["ref_grad"]
        if k in ref["ref"]
    ]
    rm = [
        (int(k), ref["ref_grad_remat"][k]["med"] / ref["ref"][k]["med"])
        for k in ref["ref_grad_remat"]
        if k in ref["ref"]
    ]
    c.add("reference: stored residuals", sorted(rr), COL["ref"], marker="square")
    c.add("reference: rematerialised", sorted(rm), COL["ref"], dash="5 4", marker="square")
    for run, lab, col in [
        ("single:1x1", "padded P=1", PCOL[1]),
        ("multi:2x2", "padded P=4", PCOL[4]),
        ("multi:32x1", "padded P=32", PCOL[32]),
    ]:
        f, gg, gm = (
            g("fwd", "padded", run),
            g("grad", "padded", run),
            g("grad_remat", "padded", run),
        )
        c.add(
            f"{lab}: stored",
            sorted((int(k), gg[k]["med"] / f[k]["med"]) for k in gg if k in f),
            col,
            marker="circle",
        )
        c.add(
            f"{lab}: remat",
            sorted((int(k), gm[k]["med"] / f[k]["med"]) for k in gm if k in f),
            col,
            dash="5 4",
            marker="circle",
        )
    return c.svg()


def mem_chart():
    c = C.Chart(
        "Peak device memory of the rematerialised gradient",
        "global grid M = N",
        "GiB per GPU",
        xticks=SIZES,
        xtickfmt=lambda v: f"{int(v)}",
    )
    for run, lab, col in [("single|1x1", "P=1", PCOL[1]), ("multi|2x2", "P=4 (2×2)", PCOL[4])]:
        pts = [
            (s, mem[f"{run}|padded|grad_remat|{s}"])
            for s in SIZES
            if f"{run}|padded|grad_remat|{s}" in mem and mem[f"{run}|padded|grad_remat|{s}"] > 0.001
        ]
        c.add(lab, pts, col)
    c.hline(95.6, "GH200: 95.6 GiB")
    return c.svg()


def exch_chart():
    c = C.Chart(
        "Exchange-only time per call, padded transport",
        "global grid M = N",
        "ms per exchange (median, IQR)",
        xticks=SIZES,
        xtickfmt=lambda v: f"{int(v)}",
    )
    for run, lab in [
        ("single:1x1", "P=1"),
        ("multi:2x2", "P=4 (2×2)"),
        ("multi:8x1", "P=8"),
        ("multi:32x1", "P=32"),
    ]:
        e, eg = (
            D["exch"]["exch"]["padded"].get(run, {}),
            D["exch"]["exch_grad"]["padded"].get(run, {}),
        )
        if e:
            c.add(f"{lab} forward", ser(e), PCOL[P[run]], label=False)
        if eg:
            c.add(f"{lab} forward+adjoint", ser(eg), PCOL[P[run]], dash="5 4", label=False)
    return c.svg()


def ncu_bars():
    # from ncu_summary.md (already aggregated): hard-code the two kernel tables read from the profile
    ref_k = [
        ("subtract fusion", 6143, 87),
        ("add/divide fusion", 3754, 85),
        ("concatenate fusion (×2)", 2596, 82),
        ("multiply fusion 1", 1750, 91),
        ("concatenate fusion", 1741, 92),
        ("multiply fusion", 1448, 73),
    ]
    fwd_k = [
        ("subtract fusion", 6699, 82),
        ("pad fusion (×3)", 5532, 58),
        ("select fusion (×3)", 5508, 91),
        ("multiply fusion", 4648, 69),
        ("add/divide fusion", 3491, 92),
        ("scatter fusion (×3)", 15, 15),
    ]

    def one(title, ks):
        w, h = 900, 44 + 30 * len(ks)
        o = [
            f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{title}">',
            f'<text x="8" y="16" class="ct">{title}</text>',
        ]
        mx = max(k[1] for k in ks)
        for i, (n, us, dram) in enumerate(ks):
            y = 34 + i * 30
            bw = 340 * us / mx
            o.append(
                f'<text x="8" y="{y + 12}" class="tk">{n}</text><g class="pt"><rect x="200" y="{y}" width="{bw:.1f}" height="16" fill="var(--accent)" rx="2"/><title>{n}: {us / 1000:.2f} ms per step, DRAM {dram}% of peak</title></g>'
            )
            o.append(
                f'<text x="{200 + bw + 6:.1f}" y="{y + 12}" class="tk">{us / 1000:.2f} ms</text>'
            )
            o.append(
                f'<rect x="640" y="{y}" width="{1.6 * dram:.0f}" height="16" fill="var(--ink-3)" rx="2"/><text x="{640 + 1.6 * dram + 6:.0f}" y="{y + 12}" class="tk">{dram}%</text>'
            )
        o.append(
            f'<text x="200" y="{h - 6}" class="al">kernel time per step</text><text x="640" y="{h - 6}" class="al">DRAM throughput, % of 4 TB/s peak</text></svg>'
        )
        return "\n".join(o)

    return one("Reference model, one step at 16384², 1 GPU (ncu)", ref_k), one(
        "Padded transport, sharded forward, P=1, 16384² (ncu)", fwd_k
    )


# ---------- tables ----------
def node_table(mode, size):
    rows = []
    for run, lab in [
        ("single:1x1", "1 GPU"),
        ("multi:2x1", "2 GPUs, 2×1"),
        ("single:2x2", "4 GPUs, 2×2, one process"),
        ("multi:2x2", "4 GPUs, 2×2"),
        ("multi:4x1", "4 GPUs, 4×1"),
        ("multi:8x1", "8 GPUs, 8×1 (2 nodes)"),
        ("multi:4x2", "8 GPUs, 4×2 (2 nodes)"),
        ("multi:16x1", "16 GPUs, 16×1 (4 nodes)"),
        ("multi:4x4", "16 GPUs, 4×4 (4 nodes)"),
        ("multi:32x1", "32 GPUs, 32×1 (8 nodes)"),
        ("multi:8x4", "32 GPUs, 8×4 (8 nodes)"),
    ]:
        cells = [lab]
        for t in ["padded", "coloured2ph", "coloured8", "allgather"]:
            r = g(mode, t, run).get(str(size))
            b = med(mode, t, "single:1x1", size)
            if r and b:
                sp = b / r["med"]
                e = sp / P[run]
                cells.append(
                    f'{fmt(r["med"])} ms <span class="mut">({sp:.1f}×, {100 * e:.0f}%)</span>'
                )
            else:
                cells.append("–")
        rows.append(cells)
    return table(
        ["configuration"] + [TNAME[t] for t in ["padded", "coloured2ph", "coloured8", "allgather"]],
        rows,
    )


def baseline_table():
    rows = []
    for s in SIZES:
        k = str(s)
        r = [
            f"{s}²",
            fmt(ref["ref"].get(k, {}).get("med")),
            fmt(ref["ref_grad"].get(k, {}).get("med")),
            fmt(ref["ref_grad_remat"].get(k, {}).get("med")),
            fmt(med("fwd", "padded", "single:1x1", s)),
            fmt(med("grad", "padded", "single:1x1", s)),
            fmt(med("grad_remat", "padded", "single:1x1", s)),
            fmt(med("fwd", "allgather", "single:1x1", s)),
            fmt(med("fwd", "coloured2ph", "single:1x1", s)),
        ]
        rows.append(r)
    return table(
        [
            "grid",
            "reference fwd",
            "reference grad",
            "reference grad (remat)",
            "padded fwd",
            "padded grad",
            "padded grad (remat)",
            "all-gather fwd",
            "coloured 2-phase fwd",
        ],
        rows,
    )


def remat_table():
    rows = []
    for run, lab in [
        ("single:1x1", "P=1"),
        ("multi:2x1", "P=2"),
        ("multi:2x2", "P=4"),
        ("multi:8x1", "P=8"),
        ("multi:16x1", "P=16"),
        ("multi:32x1", "P=32"),
    ]:
        cells = [lab]
        for s in [256, 1024, 2048, 4096, 8192, 16384]:
            a, b = med("grad_remat", "padded", run, s), med("grad", "padded", run, s)
            cells.append(f"{a / b:.2f}" if a and b else "–")
        rows.append(cells)
    return table(["padded", "256²", "1024²", "2048²", "4096²", "8192²", "16384²"], rows)


def mem_table():
    rows = []
    for s in [1024, 2048, 4096, 8192, 16384]:
        r = [f"{s}²"]
        for key, oomk in [
            ("single|1x1|padded|grad_remat", "strong|single|1x1|padded|grad_remat"),
            ("multi|2x2|padded|grad_remat", "strong|multi|2x2|padded|grad_remat"),
        ]:
            v = mem.get(f"{key}|{s}")
            if v:
                r.append(f"{v:.1f} GiB")
            elif D["oom"].get(oomk) == s:
                r.append("OOM")
            else:
                r.append("–")
        for lay in ["1x1", "2x1", "2x2", "8x1"]:
            proc = "single" if lay in ("1x1", "2x1") else "multi"
            pg = D["pgrad"].get(f"{proc}|{lay}|padded|grad|{s}")
            default_oom = D["oom"].get(f"strong|{proc}|{lay}|padded|grad")
            if pg is None:
                r.append("–")
            elif isinstance(pg, str):
                r.append(f"OOM")
            else:
                r.append(
                    f"{fmt(pg)} ms"
                    + (
                        ' <span class="mut">(default allocator: OOM)</span>'
                        if default_oom == s
                        else ""
                    )
                )
        rows.append(r)
    return table(
        [
            "grid",
            "remat peak, P=1",
            "remat peak, P=4 (2×2)",
            "plain grad P=1 (prealloc)",
            "plain grad P=2",
            "plain grad P=4 (2×2)",
            "plain grad P=8 (8×1)",
        ],
        rows,
    )


def scan_table():
    S = D["scanstep"]
    rows = []
    for mode, lab in [
        ("fwd", "padded forward"),
        ("ref", "reference forward"),
        ("grad_remat", "padded grad (remat)"),
    ]:
        t = "padded" if mode != "ref" else None
        for s in [2048, 4096, 8192, 16384]:
            cells = [lab if s == 2048 else "", f"{s}²"]
            for n in [1, 2, 5, 20]:
                r = S.get(f"{n}|{t}|{mode}|{s}")
                cells.append(fmt(r["med"]) if r else "–")
            sweep = (
                med(mode, "padded", "single:1x1", s)
                if mode != "ref"
                else ref["ref"].get(str(s), {}).get("med")
            )
            r1 = S.get(f"1|{t}|{mode}|{s}")
            cells.append(f"{sweep / r1['med']:.2f}" if r1 and sweep else "–")
            rows.append(cells)
    return table(
        [
            "mode",
            "grid",
            "1 step / call",
            "2 steps",
            "5 steps",
            "20 steps",
            "20-step sweep ÷ 1 step",
        ],
        rows,
    )


def battery_table():
    B = D["battery2048"]
    rows = []
    for t in ["padded", "coloured2ph", "allgather", "ragged"]:
        b = B.get(t, {})
        failed = [
            k[:2] for k in sorted(b) if k.endswith("_pass") and k != "all_pass" and b[k] is False
        ]
        rows.append(
            [
                TNAME[t],
                "pass" if b.get("all_pass") else "FAIL " + ",".join(failed),
                f"{b.get('T4_max_abs_diff', 0):.1e}"
                if b.get("T4_max_abs_diff") is not None
                else "–",
                f"{b.get('T5_max_rel_diff_common'):.1e}"
                if b.get("T5_max_rel_diff_common") is not None
                else "–",
                f"{b.get('T6_rate2_last'):.3f}" if b.get("T6_rate2_last") is not None else "–",
            ]
        )
    return table(
        [
            "transport",
            "battery",
            "forward vs reference (max abs diff)",
            "gradient vs reference (rel)",
            "Taylor remainder order",
        ],
        rows,
    )


def lin_table():
    L = D["lin"]
    rows = []
    for t, mode, lab in [
        ("padded", "fwd", "padded forward"),
        (None, "ref", "reference forward"),
        ("padded", "grad_remat", "padded grad (remat)"),
        (None, "ref_grad_remat", "reference grad (remat)"),
    ]:
        for s in [64, 4096]:
            t10, t40 = L.get(f"10|{t}|{mode}|{s}"), L.get(f"40|{t}|{mode}|{s}")
            if t10 and t40:
                a = (t40 * 40 - t10 * 10) / 30
                c = t10 * 10 - 10 * a
                rows.append(
                    [lab, f"{s}²", fmt(a), fmt(c), f"{100 * c / (20 * a):+.0f}%" if a else "–"]
                )
    return table(
        ["mode", "grid", "asymptotic ms/step (a)", "fixed ms/call (c)", "bias of 20-step numbers"],
        rows,
    )


def exch_share_table():
    rows = []
    for run, lab in [
        ("multi:2x2", "P=4 (2×2)"),
        ("multi:8x1", "P=8"),
        ("multi:16x1", "P=16"),
        ("multi:32x1", "P=32"),
    ]:
        cells = [lab]
        for s in [2048, 4096, 8192, 16384]:
            e = D["exch"]["exch"]["padded"].get(run, {}).get(str(s))
            eg = D["exch"]["exch_grad"]["padded"].get(run, {}).get(str(s))
            f = med("fwd", "padded", run, s)
            gm = med("grad_remat", "padded", run, s)
            cells.append(
                (f"{fmt(e['med'])} ms ({100 * e['med'] / f:.0f}%)" if e and f else "–")
                + " / "
                + (f"{fmt(eg['med'])} ms ({100 * eg['med'] / gm:.0f}%)" if eg and gm else "–")
            )
        rows.append(cells)
    return table(["padded", "2048²", "4096²", "8192²", "16384²"], rows)


def iqr_stats():
    rel = []
    for mode in D["strong"]:
        for t in D["strong"][mode]:
            for run in D["strong"][mode][t]:
                for k, r in D["strong"][mode][t][run].items():
                    if r["med"] > 0 and int(k) >= 1024:
                        rel.append((r["q75"] - r["q25"]) / r["med"])
    n = len(rel)
    return n, sum(1 for x in rel if x < 0.01) / n, sum(1 for x in rel if x < 0.05) / n, max(rel)


IQR = iqr_stats()


def share(run, size, mode_e, mode_s, t="padded"):
    e = D["exch"][mode_e][t].get(run, {}).get(str(size))
    st_ = med(mode_s, t, run, size)
    return e["med"] / st_ if e and st_ else None


def strip_block_rows():
    rows = []
    for strip, block, size_f, size_g in [
        ("multi:4x1", "multi:2x2", 16384, 8192),
        ("multi:8x1", "multi:4x2", 16384, 8192),
        ("multi:16x1", "multi:4x4", 16384, 8192),
        ("multi:32x1", "multi:8x4", 16384, 8192),
    ]:
        for t in ["padded", "coloured2ph"]:
            f1, f2 = med("fwd", t, strip, size_f), med("fwd", t, block, size_f)
            g1, g2 = med("grad_remat", t, strip, size_g), med("grad_remat", t, block, size_g)
            rows.append(
                [
                    f"P={P[strip]}: {strip.split(':')[1]} vs {block.split(':')[1]}",
                    TNAME[t],
                    f"{fmt(f1)} / {fmt(f2)}" + (f" ({f1 / f2:.2f})" if f1 and f2 else ""),
                    f"{fmt(g1)} / {fmt(g2)}" + (f" ({g1 / g2:.2f})" if g1 and g2 else ""),
                ]
            )
    return rows


json.dump(
    {
        "base16": base16,
        "ref16": ref16,
        "sp_p32": sp_p32,
        "sp_c32": sp_c32,
        "sp_c8": sp_c8,
        "gr32": gr32,
        "gr1": gr1,
        "grc32": grc32,
        "grc1": grc1,
        "overhead16": overhead16,
        "ad_ref": ad_ref,
        "ad_ref_remat": ad_ref_remat,
        "remat_vs_grad_2048": remat_vs_grad_2048,
    },
    open("prose_numbers.json", "w"),
    indent=1,
)
print(
    json.dumps(
        {
            "base16": base16,
            "ref16": ref16,
            "sp_p32": sp_p32,
            "sp_c32": sp_c32,
            "sp_c8": sp_c8,
            "gr32": gr32,
            "gr1": gr1,
            "grc32": grc32,
            "grc1": grc1,
            "overhead16": overhead16,
            "ad_ref": ad_ref,
            "ad_ref_remat": ad_ref_remat,
            "remat_vs_grad_2048": remat_vs_grad_2048,
        },
        indent=1,
    )
)
