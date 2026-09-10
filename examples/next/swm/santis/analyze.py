#!/usr/bin/env python
"""Tables and plots from scaling_bench.py sweeps on Santis.

    python examples/next/swm/santis/analyze.py RESULTS_DIR [--out PLOTS_DIR]

Reads ``{strong|weak|exch|rstrong|rweak}_{single|multiN}_<jobid>.jsonl``; all files of a (prefix,
run) are merged in job-id order and deduplicated by case, last occurrence wins. ``rstrong``/``rweak``
(rematerialised gradient, mode ``grad_remat``) merge into strong/weak. A line that does not parse
(still being written) is skipped. 1x1 always comes from the
single-process run, also in "multi" columns; multi columns are ordered by P then layout and
carry the node count ``(kn)`` at 4 GPUs per node. Cells show ``min/median`` ms per step (one
number without ``samples_step_ms``); speedups and efficiencies use medians, ``+-`` half the
interval spanned by the interquartile ranges of numerator and denominator.

Sanity files (``roof``, ``lin10``/``lin40``, ``repeat``, ``prealloc``, ``battery2048``, ``smoke``) are
kept as raw rows; each section is skipped with a note while its file is missing. HBM_PEAK_GBPS is
the assumed GH200 HBM3 peak (4.0 TB/s, NVIDIA datasheet). XLA's ``bytes accessed`` counts the
while-loop body once per module, so it is not a per-step quantity and no bandwidth is derived from
it; the roofline evidence is the ncu per-kernel DRAM utilisation in the profile section.
"""

import argparse
import collections
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


LAYOUTS = ("1x1", "2x1", "2x2", "4x1", "4x2", "8x1", "4x4", "16x1", "8x4", "32x1")
P_OF = {lay: int(lay.split("x")[0]) * int(lay.split("x")[1]) for lay in LAYOUTS}
STRIP = {lay: lay.endswith("x1") for lay in LAYOUTS}
PS = (1, 2, 4, 8, 16, 32)
TRANSPORTS = ("padded", "coloured8", "coloured2ph", "allgather", "ragged")
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
COLOUR = dict(zip(TRANSPORTS, PALETTE))
PCOL = dict(zip(PS, PALETTE))
REF = "#52514e"
DASH = {"single": "-", "multi": "--"}
COMBOS = sorted(
    [("single", lay) for lay in LAYOUTS[:4]] + [("multi", lay) for lay in LAYOUTS[1:]],
    key=lambda c: (P_OF[c[1]], c[1], c[0] == "multi"),
)
LABEL = {
    (p, lay): "1x1"
    if lay == "1x1"
    else f"{p} {lay}" + (f" ({max(1, P_OF[lay] // 4)}n)" if p == "multi" else "")
    for p, lay in COMBOS
}
MODES = ("fwd", "grad", "grad_remat")
OOM_NOTE = " (non-preallocating allocator, 20 steps; see 5f)"
HBM_PEAK_GBPS = 4000.0
MAIN = {"strong": "strong", "weak": "weak", "exch": "exch", "rstrong": "strong", "rweak": "weak"}
AUX = {}  # (prefix, run) -> rows of the newest sanity file
NOTED = set()
REF_MODE = {"fwd": "ref", "grad": "ref_grad", "grad_remat": "ref_grad_remat"}
T = {}  # (kind, proc, layout, transport, mode, size) -> (min, q25, median, q75) ms/step
MEM = {}  # same key -> peak GB of the case, "?" when the cumulative peak did not rise
OOM = {}  # keys of status=oom rows


def load(results_dir):
    files = collections.defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*.jsonl")):
        m = re.match(r"([a-z0-9]+)_(single|multi\d*)_(\d+)\.jsonl$", os.path.basename(path))
        if m:
            files[m.groups()[:2]].append((int(m.group(3)), path))
    dropped = collections.defaultdict(list)
    ndup = 0
    for (prefix, run), paths in sorted(files.items()):
        paths = [path for _, path in sorted(paths)]
        print(f"using {' + '.join(os.path.basename(path) for path in paths)}")
        rows = {}
        for path in paths:
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # re-runs append to the same file or add a newer file: the last occurrence of a case wins
                rows[
                    tuple(
                        r.get(k) for k in ("layout", "transport", "mode", "size", "weak", "n_steps")
                    )
                ] = r
                ndup += 1
        ndup -= len(rows)
        rows = list(rows.values())
        if prefix not in MAIN:
            AUX[prefix, run] = rows
            continue
        for r in rows:
            key = (
                MAIN[prefix],
                "single" if run == "single" else "multi",
                r["layout"],
                r["transport"],
                r["mode"],
            )
            if r["status"] != "ok":
                dropped[(*key, r["status"])].append(r["size"])
                if r["status"] == "oom":
                    OOM[(*key, r["size"])] = True
            else:
                T[(*key, r["size"])] = stat(r)
            if r["status"] == "ok" and "peak_mem_bytes_delta" in r:
                MEM[(*key, r["size"])] = (
                    r["peak_mem_cumulative_bytes"] / 2**30 if r["peak_mem_bytes_delta"] > 0 else "?"
                )
    print(f"duplicate case rows dropped (last occurrence kept): {ndup}")
    print(
        "\n### Rows with status != ok (dropped)\n\n| kind | proc | layout | transport | mode | status | sizes |\n|---|---|---|---|---|---|---|"
    )
    for k, sizes in sorted(dropped.items(), key=str):
        note = OOM_NOTE if k[4] in ("grad", "ref_grad") and k[5] == "oom" else ""
        print(
            "| "
            + " | ".join(str(x) for x in k)
            + note
            + " | "
            + ", ".join(map(str, sorted(sizes)))
            + " |"
        )


def stat(r):
    s = r.get("samples_step_ms")
    return (min(s), *np.percentile(s, [25, 50, 75])) if s else (r["per_step_ms"],) * 4


def cases(prefix, run="single", ok=True):
    """{(layout, transport, mode, size): row} of a sanity file, or None when it is not there yet."""
    if (prefix, run) not in AUX:
        if (prefix, run) not in NOTED:
            print(f"\n({prefix}_{run}: no file yet)")
        NOTED.add((prefix, run))
        return None
    return {
        tuple(r.get(k) for k in ("layout", "transport", "mode", "size")): r
        for r in AUX[prefix, run]
        if not ok or r["status"] == "ok"
    }


def get(kind, proc, layout, transport, mode, size, d=T):
    if layout == "1x1" or transport is None:
        proc = "single"
    return d.get((kind, proc, layout, transport, mode, size))


def med(*key):
    x = get(*key)
    return None if x is None else x[2]


def ref(kind, mode, size, stat=med):
    return stat(kind, "single", "1x1", None, REF_MODE.get(mode, mode), size)


def sizes_of(kind):
    return sorted({k[5] for k in T if k[0] == kind})


def transports_of(kind):
    return [tr for tr in TRANSPORTS if any(k[0] == kind and k[3] == tr for k in T)]


def div(a, b):
    return None if a is None or b is None else a / b


def ratio(a, b):
    """(median ratio, half-width) of a/b from (min, q25, med, q75) tuples."""
    return None if a is None or b is None else (a[2] / b[2], (a[3] / b[1] - a[1] / b[3]) / 2)


def fmt(v):
    if isinstance(v, tuple):
        return f"{v[2]:.3g}" if v[0] == v[2] else f"{v[0]:.3g}/{v[2]:.3g}"
    return "" if v is None else v if isinstance(v, str) else f"{v:.3g}"


def pm(r, scale=1, digits=2):
    err = round(scale * r[1], digits)
    return f"{scale * r[0]:.{digits}f}" + (f"+-{err:.{digits}f}" if err else "")


def speedup(base, v, P):
    r = ratio(base, v)
    return None if r is None else f"{pm(r)}x, {pm(r, 100 / P, 0)}%"


def pair(a, b):
    return (
        None
        if a is None and b is None
        else " / ".join("-" if x is None else f"{x:.3g}" for x in (a, b))
    )


def table(title, first, keys, cols):
    cols = [(lab, [f(k) for k in keys]) for lab, f in cols]
    cols = [(lab, vals) for lab, vals in cols if any(v is not None for v in vals)]
    body = [[str(k)] + [fmt(vals[i]) for _, vals in cols] for i, k in enumerate(keys)]
    body = [r for r in body if any(r[1:])]
    if not body:
        return
    header = [first] + [lab for lab, _ in cols]
    print(f"\n### {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in body:
        print("| " + " | ".join(r) + " |")


def baseline():
    sizes, trs = sizes_of("strong"), transports_of("strong")

    def g(mode, tr=None):
        return lambda s: get("strong", "single", "1x1", tr, mode, s)

    def rel(m1, t1, m2, t2=None):
        return lambda s: div(
            med("strong", "single", "1x1", t1, m1, s), med("strong", "single", "1x1", t2, m2, s)
        )

    table(
        "2a. Single-GPU baseline (1x1, single process): min/median ms per step",
        "size",
        sizes,
        [(REF_MODE[m], g(REF_MODE[m])) for m in MODES]
        + [(f"{tr} {m}", g(m, tr)) for tr in trs for m in MODES],
    )
    table(
        "2a. Baseline ratios of medians: grad/fwd and sharded-1x1/ref",
        "size",
        sizes,
        [
            ("ref grad/fwd", rel("ref_grad", None, "ref")),
            ("ref grad_remat/fwd", rel("ref_grad_remat", None, "ref")),
        ]
        + [(f"{tr} {m}/fwd", rel(m, tr, "fwd", tr)) for tr in trs for m in MODES[1:]]
        + [(f"{tr} {m}/{REF_MODE[m]}", rel(m, tr, REF_MODE[m])) for tr in trs for m in MODES],
    )


def scaled(kind, proc, lay, tr, mode, s):
    v, base = get(kind, proc, lay, tr, mode, s), get(kind, "single", "1x1", tr, mode, s)
    if v is None or base is None or lay == "1x1":
        return v
    if kind == "weak":
        return f"{fmt(v)} ({pm(ratio(base, v), 100, 0)}%)"
    return f"{fmt(v)} ({speedup(base, v, P_OF[lay])})"


def scaling(kind):
    strong = kind == "strong"
    head = "2b. Strong scaling" if strong else "2c. Weak scaling"
    what = (
        "min/median ms (speedup, efficiency)" if strong else "min/median ms (efficiency)"
    ) + "; sizes <= 512 are launch-floor dominated (fixed ~0.16 ms/call)"
    for mode in MODES:
        for tr in transports_of(kind):
            cols = [("ref", lambda s, mode=mode: ref(kind, mode, s, get))] + [
                (
                    LABEL[p, lay],
                    lambda s, p=p, lay=lay, tr=tr, mode=mode: scaled(kind, p, lay, tr, mode, s),
                )
                for p, lay in COMBOS
            ]
            table(
                f"{head}, {mode}, {tr}: {what}",
                "size" if strong else "block edge",
                sizes_of(kind),
                cols,
            )


def trs_for(mode):
    # ragged's gradient fails the battery (5e): keep it for fwd only
    return [tr for tr in transports_of("strong") if tr != "ragged" or mode == "fwd"]


def common_size(mode):
    # largest size with the most (transport, layout) cells filled; with complete data this
    # is the largest size every transport completed everywhere
    trs = trs_for(mode)

    def filled(s):
        return sum(
            get("strong", p, lay, tr, mode, s) is not None for tr in trs for p, lay in COMBOS
        )

    best = max(sizes_of("strong"), key=lambda s: (filled(s), s), default=None)
    return best if best is not None and filled(best) else None


def ranking():
    print(
        "\n(2d, 2g, transports_bar: ragged excluded from grad and grad_remat, its gradient fails the battery, see 5e)"
    )
    for mode in MODES:
        trs = trs_for(mode)
        s = common_size(mode)
        if s is None:
            continue
        cols = []
        for p, lay in COMBOS:
            vals = {tr: med("strong", p, lay, tr, mode, s) for tr in trs}
            best = min((v for v in vals.values() if v is not None), default=None)
            cols.append(
                (
                    LABEL[p, lay],
                    lambda tr, vals=vals, best=best: None
                    if vals[tr] is None
                    else f"{vals[tr]:.3g} ({vals[tr] / best:.2f}x)",
                )
            )
        table(
            f"2d. Transport ranking at M=N={s}, {mode}: median ms (x fastest in column)",
            "transport",
            trs,
            cols,
        )


def mode_ratio(p, lay, tr, s, m1="grad", m2="fwd"):
    return div(med("strong", p, lay, tr, m1, s), med("strong", p, lay, tr, m2, s))


def ad_cost():
    for title, m1, m2 in (
        ("2e. Cost of AD (grad/fwd, medians)", "grad", "fwd"),
        ("2e. Cost of AD (grad_remat/fwd, medians)", "grad_remat", "fwd"),
        ("2h. Remat overhead (grad_remat/grad, medians)", "grad_remat", "grad"),
    ):
        pg = pgrad_rows()
        for tr in transports_of("strong"):
            cols = [
                (
                    "ref",
                    lambda s, m1=m1, m2=m2: mode_ratio(
                        "single", "1x1", None, s, REF_MODE[m1], REF_MODE[m2]
                    ),
                )
            ]
            cols += [
                (
                    LABEL[p, lay],
                    lambda s, p=p, lay=lay, tr=tr, m1=m1, m2=m2: mode_ratio(p, lay, tr, s, m1, m2),
                )
                for p, lay in COMBOS
            ]
            if (m1, m2, tr) == ("grad", "fwd", "padded"):
                cols += [
                    (
                        f"{LABEL[p, lay]} grad (prealloc)/fwd",
                        lambda s, p=p, lay=lay, pg=pg: div(
                            stat(r)[2]
                            if (r := pg.get((p, lay, "padded", "grad", s))) and r["status"] == "ok"
                            else None,
                            med("strong", p, lay, "padded", "fwd", s),
                        ),
                    )
                    for p, lay in COMBOS
                ]
            table(f"{title}, {tr}", "size", sizes_of("strong"), cols)


def p4_speedup():
    def cell(tr, lay, mode, s):
        base = get("strong", "single", "1x1", tr, mode, s)
        sp = [ratio(base, get("strong", p, lay, tr, mode, s)) for p in ("single", "multi")]
        return (
            None
            if sp == [None, None]
            else " / ".join("-" if r is None else pm(r) + "x" for r in sp)
        )

    for mode in MODES[1:]:
        table(
            f"2f. {mode} speedup at P=4 vs 1x1: single / multi process",
            "size",
            sizes_of("strong"),
            [
                (f"{tr} {lay}", lambda s, tr=tr, lay=lay, mode=mode: cell(tr, lay, mode, s))
                for tr in transports_of("strong")
                for lay in ("2x2", "4x1")
            ],
        )


def node_scaling():
    sizes = sizes_of("strong")
    combos = [c for c in COMBOS if P_OF[c[1]] != 2]
    for mode in MODES:
        for tr in trs_for(mode):

            def have(P, s, tr=tr, mode=mode):
                return any(
                    get("strong", p, lay, tr, mode, s) for p, lay in combos if P_OF[lay] == P
                )

            ps = [P for P in PS if P != 2 and any(have(P, s) for s in sizes)]
            ok = [s for s in sizes if all(have(P, s) for P in ps)]
            if len(ps) < 2 or not ok:
                continue
            s = ok[-1]
            vals = {LABEL[c]: (c, get("strong", *c, tr, mode, s)) for c in combos}
            b1 = vals["1x1"][1]
            b4 = min(
                (v for (p, lay), v in vals.values() if v and p == "multi" and P_OF[lay] == 4),
                key=lambda v: v[2],
                default=None,
            )

            def col(f, vals=vals):
                return lambda k: None if vals[k][1] is None else f(P_OF[vals[k][0][1]], vals[k][1])

            table(
                f"2g. Node scaling, {mode}, {tr}, M=N={s} (4 GPUs per node)",
                "run",
                list(vals),
                [
                    ("P", col(lambda P, v: str(P))),
                    ("nodes", col(lambda P, v: str(max(1, P // 4)))),
                    ("min/median ms", col(lambda P, v: v)),
                    ("vs 1x1: speedup, eff", col(lambda P, v, b1=b1: speedup(b1, v, P))),
                    (
                        "vs P=4 multi: speedup, eff",
                        col(lambda P, v, b4=b4: None if P == 1 else speedup(b4, v, P / 4)),
                    ),
                ],
            )


STRIP_PAIRS = (("4x1", "2x2"), ("8x1", "4x2"), ("16x1", "4x4"), ("32x1", "8x4"))


def strip_vs_block():
    def cell(tr, mode, strip, block):
        sizes = [
            s
            for s in sizes_of("strong")
            if med("strong", "multi", strip, tr, mode, s)
            and med("strong", "multi", block, tr, mode, s)
        ]
        if not sizes:
            return None
        a, b = (med("strong", "multi", lay, tr, mode, sizes[-1]) for lay in (strip, block))
        return f"M=N={sizes[-1]}: {a:.3g} / {b:.3g} = {a / b:.2f}"

    table(
        "2i. 1-D strip vs 2-D block at the same P (multi-process, largest common size): median ms strip / block = ratio",
        "P: strip vs block",
        [f"P={P_OF[a]}: {a} vs {b}" for a, b in STRIP_PAIRS],
        [
            (
                f"{tr} {mode}",
                lambda k, tr=tr, mode=mode: cell(
                    tr,
                    mode,
                    *STRIP_PAIRS[[f"P={P_OF[a]}: {a} vs {b}" for a, b in STRIP_PAIRS].index(k)],
                ),
            )
            for tr in ("padded", "coloured2ph")
            for mode in ("fwd", "grad_remat")
        ],
    )


def memory():
    sizes = sizes_of("strong")
    trs = [tr for tr in transports_of("strong") if any(k[3] == tr for k in MEM)]
    p4 = next((c for c in COMBOS if P_OF[c[1]] == 4 and any(k[1:3] == c for k in MEM)), None)

    pg = {
        k: r["peak_mem_cumulative_bytes"] / 2**30
        for k, r in pgrad_rows().items()
        if r["status"] == "ok" and r.get("peak_mem_bytes_delta", 0) > 0
    }

    def cell(p, lay, tr, mode, s):
        parts = []
        if get("strong", p, lay, tr, mode, s, OOM):
            parts.append("OOM" + (OOM_NOTE if mode in ("grad", "ref_grad") else ""))
        elif (v := get("strong", p, lay, tr, mode, s, MEM)) is not None:
            parts.append(fmt(v))
        if (v := pg.get(("single" if lay == "1x1" else p, lay, tr, mode, s))) is not None:
            parts.append(f"{v:.3g} (prealloc)")
        return "; ".join(parts) or None

    for tr in trs:
        combos = [("single", "1x1")] + ([p4] if p4 else [])
        table(
            f"4. Peak device memory per GPU (GB), {tr}; '?' = cumulative peak did not rise",
            "size",
            sizes,
            [
                (f"{REF_MODE[m]}", lambda s, m=m: cell("single", "1x1", None, REF_MODE[m], s))
                for m in MODES
            ]
            + [
                (f"{m} {LABEL[c]}", lambda s, c=c, m=m, tr=tr: cell(*c, tr, m, s))
                for m in MODES
                for c in combos
            ],
        )


def exchange():
    sizes = sizes_of("exch")
    for tr in transports_of("exch"):
        table(
            f"3a. Exchange time, {tr}: median ms per call, exch / exch_grad",
            "size",
            sizes,
            [
                (
                    LABEL[c],
                    lambda s, c=c, tr=tr: pair(
                        med("exch", *c, tr, "exch", s), med("exch", *c, tr, "exch_grad", s)
                    ),
                )
                for c in COMBOS
            ],
        )
    for tr in ("padded", "allgather"):
        table(
            f"3b. Exchange share of a step, {tr}: exch/fwd / exch_grad/grad (medians); a standalone exchange call carries a "
            "0.15-0.4 ms dispatch floor, so the share is only meaningful for M=N >= 4096 (smaller sizes blanked)",
            "size",
            sizes,
            [
                (
                    LABEL[c],
                    lambda s, c=c, tr=tr: None
                    if s < 4096
                    else pair(
                        div(med("exch", *c, tr, "exch", s), med("strong", *c, tr, "fwd", s)),
                        div(med("exch", *c, tr, "exch_grad", s), med("strong", *c, tr, "grad", s)),
                    ),
                )
                for c in COMBOS
            ],
        )


ROOF_MODES = ("fwd", "grad", "grad_remat", "ref", "ref_grad_remat")
MCOL = dict(zip(ROOF_MODES, PALETTE))


def roofline(out):
    rf = cases("roof")
    if not rf:
        return
    sizes = sorted({k[3] for k in rf})
    cols = sorted({(k[1], k[0]) for k in rf}, key=lambda c: (c[0] or "", c[1]))

    print(
        "\n5a. XLA's bytes-accessed estimate counts the while-loop body once per module (identical for n_steps 10 and 40), "
        "so no per-step bandwidth is derived from it. The roofline evidence is the ncu per-kernel DRAM utilisation in section 6."
    )
    for mode in ROOF_MODES:
        table(
            f"5a. {mode}: xla_kernels = launch-type HLO instructions in the module (body once)",
            "size",
            sizes,
            [
                (
                    f"{tr or 'ref'} {lay}",
                    lambda s, tr=tr, lay=lay, mode=mode: (r := rf.get((lay, tr, mode, s)))
                    and str(r["xla_kernels"]),
                )
                for tr, lay in cols
            ],
        )
    tr = "padded" if any(k[1] == "padded" for k in rf) else next((k[1] for k in rf if k[1]), None)
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in ROOF_MODES:
        for lay in ("1x1", "2x2"):
            rs = [rf.get((lay, None if mode.startswith("ref") else tr, mode, s)) for s in sizes]
            series(
                ax,
                sizes,
                [r and stat(r) for r in rs],
                color=MCOL[mode],
                ls=DASH["single" if lay == "1x1" else "multi"],
                marker="o",
                label=f"{mode} {lay}",
            )
            floor = [r["per_step_ms"] for r in rs[:3] if r]
            if floor and lay == "1x1":
                ax.axhline(np.median(floor), color=MCOL[mode], lw=0.8, ls="--")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="global edge M = N",
        ylabel="ms / step",
        title=f"Launch floor ({tr}; dashed: small-size plateau)",
    )
    save(fig, out, "roofline.png")


def linearity():
    l10, l40 = cases("lin10"), cases("lin40")
    if not l10 or not l40:
        return
    keys = sorted(set(l10) & set(l40), key=str)
    fit = {}
    for k in keys:
        t10, t40 = stat(l10[k])[2] * 10, stat(l40[k])[2] * 40
        a = (t40 - t10) / 30
        fit[k] = (a, t10 - 10 * a)
    table(
        "5b. Step-count linearity: t_call = a*n_steps + c from n_steps=10 and 40",
        "(layout, transport, mode, size)",
        keys,
        [
            ("a ms/step", lambda k: fit[k][0]),
            ("c ms/call", lambda k: fit[k][1]),
            ("c/(20a) bias of 20-step numbers", lambda k: fit[k][1] / (20 * fit[k][0])),
        ],
    )


def repeatability():
    for prefix, run, proc, per_case in (
        ("roof", "single", "single", False),
        ("smoke", "single", "single", False),
        ("repeat", "multi32", "multi", True),
    ):
        rows = cases(prefix, run)
        if not rows:
            continue
        pairs = {
            k: (stat(r)[2], base[2]) for k, r in rows.items() if (base := get("strong", proc, *k))
        }
        if per_case:
            table(
                f"5c. Repeatability {prefix}_{run} vs strong: median ms",
                "(layout, transport, mode, size)",
                sorted(pairs, key=str),
                [
                    (prefix, lambda k, pairs=pairs: pairs[k][0]),
                    ("strong", lambda k, pairs=pairs: pairs[k][1]),
                    ("ratio", lambda k, pairs=pairs: pairs[k][0] / pairs[k][1]),
                ],
            )
            continue
        groups = collections.defaultdict(list)
        for (lay, _tr, mode, _s), (a, b) in pairs.items():
            groups[f"{mode} {lay}"].append(a / b)
        groups["all"] = [x for g in groups.values() for x in g]
        what = (
            "n_steps sensitivity (smoke: 10 steps vs strong: 20 steps), not a repeatability check"
            if prefix == "smoke"
            else f"Repeatability {prefix}_{run} vs strong_single"
        )
        table(
            f"5c. {what}: median ratio {prefix}/strong over common cases",
            "group",
            list(groups),
            [
                ("n", lambda g, groups=groups: str(len(groups[g]))),
                ("median ratio", lambda g, groups=groups: float(np.median(groups[g]))),
                (
                    "max |ratio-1|",
                    lambda g, groups=groups: float(max(abs(x - 1) for x in groups[g])),
                ),
            ],
        )


def pgrad_rows():
    return {
        ("single" if run == "single" else "multi", *k): r
        for (prefix, run), rows in AUX.items()
        if prefix == "pgrad"
        for r in rows
        for k in [tuple(r.get(x) for x in ("layout", "transport", "mode", "size"))]
    }


def pgrad():
    rows = pgrad_rows()
    if not rows:
        print("\n(pgrad_*: no file yet)")
        return

    def cell(proc, lay, tr, mode, s):
        r = rows.get((proc, lay, tr, mode, s))
        if r is None:
            return None
        remat = med("strong", proc, lay, tr, mode + "_remat", s)
        default = (
            "ok"
            if get("strong", proc, lay, tr, mode, s)
            else "oom"
            if get("strong", proc, lay, tr, mode, s, OOM)
            else "-"
        )
        if r["status"] != "ok":
            return f"OOM (default {default}, remat {fmt(remat) or '-'})"
        g = stat(r)[2]
        return f"{g:.3g} (default {default}, remat {fmt(remat) or '-'}, remat/prealloc {fmt(div(remat, g)) or '-'})"

    for proc, lay in sorted(
        {k[:2] for k in rows}, key=lambda c: (P_OF[c[1]], c[1], c[0] == "multi")
    ):
        sizes = sorted({k[4] for k in rows if k[:2] == (proc, lay)})
        cols = sorted({(k[3], k[2]) for k in rows if k[:2] == (proc, lay)}, key=str)
        table(
            f"5f. Plain grad with XLA preallocation, {LABEL[proc, lay]}: median ms (default-allocator status, grad_remat median, ratio)",
            "size",
            sizes,
            [
                (
                    f"{mode} {tr or 'ref'}",
                    lambda s, mode=mode, tr=tr, proc=proc, lay=lay: cell(proc, lay, tr, mode, s),
                )
                for mode, tr in cols
            ],
        )


def prealloc():
    rows = cases("prealloc", ok=False)
    if not rows:
        return
    sizes = sorted({k[3] for k in rows})
    cols = sorted({(k[2], k[1]) for k in rows}, key=str)

    def default(k):
        return (
            "ok"
            if get("strong", "single", *k)
            else "oom"
            if get("strong", "single", *k, OOM)
            else "-"
        )

    table(
        "5d. Allocator sensitivity on 1x1: XLA preallocation on / default allocator (ok|oom)",
        "size",
        sizes,
        [
            (
                f"{mode} {tr or 'ref'}",
                lambda s, mode=mode, tr=tr: (r := rows.get(("1x1", tr, mode, s)))
                and f"{r['status']} / {default(('1x1', tr, mode, s))}",
            )
            for mode, tr in cols
        ],
    )


def scanstep():
    if cases("scanstep") is None:
        return
    t = {
        (r["mode"], r["size"], r["n_steps"]): stat(r)[2]
        for r in AUX["scanstep", "single"]
        if r["status"] == "ok"
    }
    steps = sorted({k[2] for k in t})
    for mode in sorted({k[0] for k in t}):
        sizes = sorted({k[1] for k in t if k[0] == mode})
        table(
            f"5g. Unrolled vs while-loop step, {mode}: median ms per step by n_steps (XLA unrolls 1-2 steps; 5 and 20 run as a while loop)",
            "size",
            sizes,
            [
                (
                    f"{'unrolled' if n <= 2 else 'while loop'} n_steps={n}",
                    lambda s, n=n, mode=mode: t.get((mode, s, n)),
                )
                for n in steps
            ]
            + [
                ("t(20)/t(1)", lambda s, mode=mode: div(t.get((mode, s, 20)), t.get((mode, s, 1)))),
                (
                    "sweep (20 steps)",
                    lambda s, mode=mode: med(
                        "strong",
                        "single",
                        "1x1",
                        None if mode.startswith("ref") else "padded",
                        mode,
                        s,
                    ),
                ),
                (
                    "sweep/t(1)",
                    lambda s, mode=mode: div(
                        med(
                            "strong",
                            "single",
                            "1x1",
                            None if mode.startswith("ref") else "padded",
                            mode,
                            s,
                        ),
                        t.get((mode, s, 1)),
                    ),
                ),
            ],
        )


def profile(results_dir):
    path = os.path.join(results_dir, "profile", "ncu_summary.md")
    if not os.path.exists(path):
        print("\n(profile/ncu_summary.md: no file yet)")
        return
    print(f"\n### 6. ncu per-kernel DRAM throughput ({path})\n\n" + open(path).read().strip())


def battery():
    if cases("battery2048", "multi32", ok=False) is None:
        return
    rows = {r["transport"]: r for r in AUX["battery2048", "multi32"]}

    def flags(r, suffix, good):
        bad = [
            f"{k[:2]}" + ("" if v is False else f"={v}")
            for k, v in r.items()
            if k[0] == "T" and k.endswith(suffix) and v != good
        ]
        return ", ".join(bad) or "all ok"

    table(
        "5e. Correctness battery at 2048x2048 per GPU on 32 GPUs",
        "transport",
        list(rows),
        [
            ("all_pass", lambda t: str(rows[t].get("all_pass"))),
            ("failed T*_pass", lambda t: flags(rows[t], "_pass", True)),
            ("T*_status != ok", lambda t: flags(rows[t], "_status", "ok")),
            ("T5_max_rel_diff_common", lambda t: rows[t].get("T5_max_rel_diff_common")),
            ("T6_rate2_last", lambda t: rows[t].get("T6_rate2_last")),
            ("max_ulp", lambda t: rows[t].get("T4_max_ulp", rows[t].get("T3_max_ulp"))),
        ],
    )


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#c3c2b7",
        "text.color": "#0b0b0b",
        "axes.labelcolor": REF,
        "xtick.color": REF,
        "ytick.color": REF,
        "legend.frameon": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)


def series(ax, xs, ys, **kw):
    pts = [(x, y if isinstance(y, tuple) else (y,) * 4) for x, y in zip(xs, ys) if y is not None]
    if pts:
        xs, ys = zip(*pts)
        err = [[y[2] - y[1] for y in ys], [y[3] - y[2] for y in ys]]
        ax.errorbar(xs, [y[2] for y in ys], yerr=err, lw=1.8, ms=4, **kw)


def save(fig, out, name):
    handles = {}
    for ax in fig.axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.5 / fig.get_figheight()),
    )
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def panels(out, name, title, ylabel, kind, rows, yscale):
    """One column per transport, x = global size, colour = P; ``rows`` = [(label, value(p, lay, tr, s), ref(s))]."""
    sizes, trs = sizes_of(kind), transports_of(kind)
    if not sizes:
        return
    fig, axes = plt.subplots(
        len(rows),
        len(trs),
        figsize=(3.4 * len(trs), 3.3 * len(rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, (row, value, ref_value) in enumerate(rows):
        for ax, tr in zip(axes[i], trs):
            ax.set(
                xscale="log",
                yscale=yscale,
                title=f"{row}, {tr}",
                xlim=(min(sizes) / 1.5, max(sizes) * 1.5),
            )
            series(ax, sizes, [ref_value(s) for s in sizes], color=REF, label="reference (1 GPU)")
            for p, lay in COMBOS:
                series(
                    ax,
                    sizes,
                    [value(p, lay, tr, s) for s in sizes],
                    color=PCOL[P_OF[lay]],
                    ls=DASH[p],
                    marker="o" if STRIP[lay] else "s",
                    label=f"P={P_OF[lay]} {lay} ({p})",
                )
        axes[i, 0].set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel("global edge M = N")
    fig.suptitle(title, y=1.0)
    save(fig, out, name)


def plot_strong(out):
    rows = [
        (
            mode,
            lambda p, lay, tr, s, mode=mode: get("strong", p, lay, tr, mode, s),
            lambda s, mode=mode: ref("strong", mode, s, get),
        )
        for mode in MODES
    ]
    panels(
        out,
        "strong_scaling.png",
        "Strong scaling: median ms per step vs problem size (IQR bars)",
        "ms / step",
        "strong",
        rows,
        "log",
    )
    rows = [
        (
            f"{m}/fwd",
            lambda p, lay, tr, s, m=m: mode_ratio(p, lay, tr, s, m),
            lambda s, m=m: mode_ratio("single", "1x1", None, s, REF_MODE[m], "ref"),
        )
        for m in MODES[1:]
    ]
    panels(
        out,
        "grad_over_fwd.png",
        "Cost of AD: grad/fwd ratio of medians vs problem size",
        "grad / fwd time",
        "strong",
        rows,
        "linear",
    )
    rows = [
        (mode, lambda p, lay, tr, s, mode=mode: get("exch", p, lay, tr, mode, s), lambda s: None)
        for mode in ("exch", "exch_grad")
    ]
    panels(
        out,
        "exchange.png",
        "Halo exchange only: median ms per call vs problem size (IQR bars)",
        "ms / call",
        "exch",
        rows,
        "log",
    )


def plot_weak(out):
    sizes, trs = sizes_of("weak"), transports_of("weak")
    if not sizes:
        return
    fig, axes = plt.subplots(
        len(MODES),
        len(sizes),
        figsize=(2.2 * len(sizes) + 1, 3 * len(MODES)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, mode in enumerate(MODES):
        for ax, s in zip(axes[i], sizes):
            ax.set_xscale("log", base=2)
            ax.set(
                xticks=PS, xticklabels=PS, title=f"{mode}, block {s}x{s}", xlabel="P" if i else None
            )
            ax.axhline(1, color=REF, lw=0.8)
            for tr in trs:
                base = get("weak", "single", "1x1", tr, mode, s)
                for p in ("single", "multi"):
                    for strip in (True, False):
                        lays = [
                            lay
                            for lay in LAYOUTS
                            if lay == "1x1" or (STRIP[lay] == strip and (p, lay) in COMBOS)
                        ]
                        ys = [ratio(base, get("weak", p, lay, tr, mode, s)) for lay in lays]
                        ys = [
                            None if r is None else (0, r[0] - r[1], r[0], r[0] + r[1]) for r in ys
                        ]
                        series(
                            ax,
                            [P_OF[lay] for lay in lays],
                            ys,
                            color=COLOUR[tr],
                            ls=DASH[p],
                            marker="o" if strip else "s",
                            label=f"{tr} ({p})",
                        )
        axes[i, 0].set_ylabel("efficiency t(1x1) / t(layout)")
    fig.suptitle(
        "Weak scaling efficiency vs P (circles: Rx x 1 layouts, squares: 2-D blocks)", y=1.0
    )
    save(fig, out, "weak_efficiency.png")


def plot_bars(out):
    trs = transports_of("strong")
    if not trs:
        return
    fig, axes = plt.subplots(len(MODES), 1, figsize=(18, 4 * len(MODES)))
    for ax, mode in zip(axes, MODES):
        trs = trs_for(mode)
        s = common_size(mode)
        if s is None:
            ax.set_axis_off()
            continue
        combos = [c for c in COMBOS if any(get("strong", *c, tr, mode, s) for tr in trs)]
        w = 0.8 / len(trs)
        for k, tr in enumerate(trs):
            v = [get("strong", p, lay, tr, mode, s) or (np.nan,) * 4 for p, lay in combos]
            ax.bar(
                np.arange(len(combos)) + (k - (len(trs) - 1) / 2) * w,
                [y[2] for y in v],
                yerr=[[y[2] - y[1] for y in v], [y[3] - y[2] for y in v]],
                width=0.9 * w,
                color=COLOUR[tr],
                label=tr,
                ecolor=REF,
            )
        ax.set(
            xticks=range(len(combos)),
            xticklabels=[LABEL[c].replace(" ", "\n", 1) for c in combos],
            title=f"{mode}, M = N = {s}",
            ylabel="ms / step",
        )
        ax.grid(axis="x", visible=False)
    fig.suptitle("Transports at the largest common size (median, IQR bars)", y=1.02)
    save(fig, out, "transports_bar.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir")
    ap.add_argument("--out", help="plot directory (default RESULTS_DIR/plots)")
    a = ap.parse_args()
    load(a.results_dir)
    baseline()
    scaling("strong")
    scaling("weak")
    ranking()
    ad_cost()
    p4_speedup()
    node_scaling()
    strip_vs_block()
    memory()
    exchange()
    out = a.out or os.path.join(a.results_dir, "plots")
    os.makedirs(out, exist_ok=True)
    for plot in (plot_strong, plot_weak, plot_bars, roofline):
        plot(out)
    linearity()
    repeatability()
    prealloc()
    pgrad()
    scanstep()
    battery()
    profile(a.results_dir)
    print(f"\nplots -> {out}")


if __name__ == "__main__":
    main()
