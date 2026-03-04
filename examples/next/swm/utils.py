# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None

FRAME_DIR = os.path.join(os.path.dirname(__file__), "frames")
_frame_data = []  # Collect (u, v, p, title) tuples for animation


def _to_2d(x):
    if x.ndim == 3:
        assert x.shape[2] == 1
        return x[:, :, 0]
    return x


def read_uvp(step, suffix, M, N):
    u_file = f"ref/{M}x{N}/u.step{step}.{suffix}.bin"
    v_file = f"ref/{M}x{N}/v.step{step}.{suffix}.bin"
    p_file = f"ref/{M}x{N}/p.step{step}.{suffix}.bin"
    u = np.fromfile(u_file).reshape(M + 1, N + 1)
    v = np.fromfile(v_file).reshape(M + 1, N + 1)
    p = np.fromfile(p_file).reshape(M + 1, N + 1)
    return u, v, p


def read_cucvzh(step, suffix, M, N):
    cu_file = f"ref/{M}x{N}/cu.step{step}.{suffix}.bin"
    cv_file = f"ref/{M}x{N}/cv.step{step}.{suffix}.bin"
    z_file = f"ref/{M}x{N}/z.step{step}.{suffix}.bin"
    h_file = f"ref/{M}x{N}/h.step{step}.{suffix}.bin"
    cu = np.fromfile(cu_file).reshape(M + 1, N + 1)
    cv = np.fromfile(cv_file).reshape(M + 1, N + 1)
    z = np.fromfile(z_file).reshape(M + 1, N + 1)
    h = np.fromfile(h_file).reshape(M + 1, N + 1)
    return cu, cv, z, h


def validate_uvp(u, v, p, M, N, step, suffix):
    u, v, p = _to_2d(u), _to_2d(v), _to_2d(p)

    u_ref, v_ref, p_ref = read_uvp(step, suffix, M, N)
    np.testing.assert_allclose(u, u_ref)
    np.testing.assert_allclose(v, v_ref)
    np.testing.assert_allclose(p, p_ref)
    print(f"step {step} {suffix} values are correct.")


def validate_cucvzh(cu, cv, z, h, M, N, step, suffix):
    cu, cv, z, h = _to_2d(cu), _to_2d(cv), _to_2d(z), _to_2d(h)

    cu_ref, cv_ref, z_ref, h_ref = read_cucvzh(step, suffix, M, N)
    np.testing.assert_allclose(cu, cu_ref)
    np.testing.assert_allclose(cv, cv_ref)
    np.testing.assert_allclose(z, z_ref)
    np.testing.assert_allclose(h, h_ref)
    print(f"step {step} {suffix} values are correct.")


def live_plot_val(fu, fv, fp, title=""):
    mxu = fu.max()
    mxv = fv.max()
    mxp = fp.max()
    clear_output(wait=True)
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

    pos1 = ax1.imshow(fp, cmap="Blues", vmin=-mxp, vmax=mxp, interpolation="none")
    ax1.set_title("p")
    plt.colorbar(pos1, ax=ax1)
    pos2 = ax2.imshow(fu, cmap="Reds", vmin=-mxu, vmax=mxu, interpolation="none")
    ax2.set_title("u")
    plt.colorbar(pos2, ax=ax2)
    pos3 = ax3.imshow(fv, cmap="Greens", vmin=-mxv, vmax=mxv, interpolation="none")
    ax3.set_title("v")
    plt.colorbar(pos3, ax=ax3)

    fig.suptitle(title)
    plt.show()


def live_plot3(fu, fv, fp, title=""):
    _frame_data.append((fu.copy(), fv.copy(), fp.copy(), title))
    print(f"  [VIS] Captured frame: {title}")


def create_animation(output_path="swm_animation.mp4", fps=20):
    """Create an MP4 animation from captured frames."""
    if not _frame_data:
        print("No frames captured.")
        return

    print(f"Creating animation from {len(_frame_data)} frames...")

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3.5), ncols=3)
    fig.subplots_adjust(top=0.85, bottom=0.05, wspace=0.3)

    fp0 = _frame_data[0][2]
    fu0 = _frame_data[0][0]
    fv0 = _frame_data[0][1]

    im1 = ax1.imshow(fp0, cmap="Blues", vmin=49999, vmax=50001, interpolation="none")
    ax1.set_title("p (pressure)")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(fu0, cmap="Reds", vmin=-1, vmax=1, interpolation="none")
    ax2.set_title("u (zonal velocity)")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    im3 = ax3.imshow(fv0, cmap="Greens", vmin=-1, vmax=1, interpolation="none")
    ax3.set_title("v (meridional velocity)")
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    suptitle = fig.suptitle(_frame_data[0][3], fontsize=14, fontweight="bold")

    def update(frame_idx):
        fu, fv, fp, title = _frame_data[frame_idx]
        im1.set_data(fp)
        im2.set_data(fu)
        im3.set_data(fv)
        suptitle.set_text(title)
        return [im1, im2, im3, suptitle]

    anim = animation.FuncAnimation(
        fig, update, frames=len(_frame_data), interval=1000 // fps, blit=True
    )

    # Use pillow for GIF (no ffmpeg dependency)
    if output_path.endswith(".mp4"):
        output_path = output_path.replace(".mp4", ".gif")
    anim.save(output_path, writer="pillow", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Animation saved to: {output_path}")
    print(f"  ({len(_frame_data)} frames, {fps} fps, ~{len(_frame_data)/fps:.1f}s duration)")


def final_validation(u, v, p, ITMAX, M, N):
    u, v, p = _to_2d(u), _to_2d(v), _to_2d(p)

    uref, vref, pref = read_uvp(ITMAX, "final", M, N)

    uval = uref - u
    vval = vref - v
    pval = pref - p

    uLinfN = np.linalg.norm(uval, np.inf)
    vLinfN = np.linalg.norm(vval, np.inf)
    pLinfN = np.linalg.norm(pval, np.inf)

    # live_plot_val(uval, vval, pval, "Val")
    print("uLinfN: ", uLinfN)
    print("vLinfN: ", vLinfN)
    print("pLinfN: ", pLinfN)
    print("udiff max: ", uval.max())
    print("vdiff max: ", vval.max())
    print("pdiff max: ", pval.max())
