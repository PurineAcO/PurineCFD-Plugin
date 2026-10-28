from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.linewidth": 1.1,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

C_OURS = "#c0392b"
C_CLIP = "#8e8e8e"
C_SA = "#1f4e79"
C_EXP = "#1e8449"
C_DES = "#6c3483"
C_UPPER = C_OURS
C_LOWER = C_SA
C_NEUTRAL = "#555555"

GAMMA = 1.4
R_AIR = 287.05

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("D:/PurineCFD-Plugin/data")
DEFAULT_CASE = DATA_ROOT / "61" / "OAT15A"


def default_outdir(case, data_root):
    try:
        rel = Path(case).resolve().relative_to(Path(data_root).resolve())
    except ValueError:
        rel = Path(case).name
    return REPO_ROOT / "results" / rel


def read_history(path):
    h = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    out = {n: h[n].astype(float) for n in h.dtype.names if n != "phase"}
    out["phase"] = np.asarray(h["phase"]).astype(str)
    return out


def read_probe(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def read_wall_cp(path):
    times = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            m = re.search(r"\bt\s+([-+0-9.eE]+)", line)
            if m:
                times.append(float(m.group(1)))
    data = np.loadtxt(path)
    if len(times) == 0 or data.size % (3 * len(times)) != 0:
        raise ValueError(f"{path} time blocks do not match data rows")
    nnode = data.shape[0] // len(times)
    return np.asarray(times), data.reshape(len(times), nnode, 3)


def read_cp_mean(path):
    out = {}
    side = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            m = re.search(r"side\s+(\w+)", s)
            if m:
                side = m.group(1)
                out.setdefault(side, [])
            continue
        if side is not None:
            out[side].append([float(v) for v in s.split()])
    return {k: np.asarray(v) for k, v in out.items() if v}


def amplitude_spectrum(t, y, nfft_x=8):
    dt = float(np.median(np.diff(t)))
    y = np.asarray(y, float)
    y = y - y.mean()
    w = np.hanning(y.size)
    nfft = int(2 ** np.ceil(np.log2(y.size * nfft_x)))
    spec = np.fft.rfft(y * w, nfft) / (0.5 * w.sum())
    return np.fft.rfftfreq(nfft, dt), np.abs(spec)


def spectral_peak(t, y, fmin, fmax):
    freqs, amp = amplitude_spectrum(t, y)
    sel = (freqs >= fmin) & (freqs <= fmax)
    f_coarse = float(freqs[sel][np.argmax(amp[sel])])
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    best = None
    for f in np.arange(f_coarse - 3.0, f_coarse + 3.0 + 1e-12, 0.002):
        a = np.column_stack(
            [np.ones_like(t), np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
        )
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        ss = float(np.sum((y - a @ coef) ** 2))
        if best is None or ss < best[0]:
            best = (ss, f, coef)
    ss, f0, coef = best
    r2 = 1.0 - ss / float(np.sum((y - y.mean()) ** 2))
    return f0, coef, r2, (freqs, amp)


def peak_time(coef, f0):
    return float(np.arctan2(coef[2], coef[1]) / (2 * np.pi * f0))


def phase_average(cp, t, f0, t_ref, nbins):
    phi = np.mod((t - t_ref) * f0, 1.0)
    idx = np.floor(phi * nbins).astype(int) % nbins
    cpb = np.zeros((nbins, cp.shape[1]))
    counts = np.zeros(nbins, int)
    for k in range(nbins):
        m = idx == k
        counts[k] = int(m.sum())
        cpb[k] = cp[m].mean(axis=0) if counts[k] else np.nan
    return (np.arange(nbins) + 0.5) / nbins, cpb, counts


def periodic_eval(centers, values, targets):
    xs = np.concatenate([centers - 1.0, centers, centers + 1.0])
    ys = np.concatenate([values, values, values], axis=0)
    return CubicSpline(xs, ys, axis=0)(np.asarray(targets, float))


def split_sides(x, y, jump_tol=0.01):
    chord = float(np.max(x))
    jumps = np.hypot(np.diff(x), np.diff(y)) > jump_tol * chord
    starts = np.r_[0, np.where(jumps)[0] + 1]
    ends = np.r_[np.where(jumps)[0], x.size - 1]
    upper = np.zeros(x.size, bool)
    lower = np.zeros(x.size, bool)
    for s, e in zip(starts, ends):
        sl = slice(s, e + 1)
        ymid = np.median(y[sl])
        if ymid > jump_tol * chord:
            upper[sl] = True
        elif ymid < -jump_tol * chord:
            lower[sl] = True
        else:
            loc_up = y[sl] > 0.0
            upper[sl] = loc_up
            lower[sl] = ~loc_up
    return upper, lower, chord


def farfield_q(case):
    steps = sorted(Path(case).glob("step_*.dat"))
    if not steps:
        raise FileNotFoundError(f"no step_*.dat under {case}, pass --q")
    d = np.loadtxt(steps[-1], skiprows=2)
    far = np.argsort(np.hypot(d[:, 0], d[:, 1]))[-50:]
    p_inf = float(np.median(d[far, 6]))
    t_inf = float(np.median(d[far, 5]))
    ma_inf = float(np.median(d[far, 7]))
    rho_inf = p_inf / (R_AIR * t_inf)
    v_inf = ma_inf * np.sqrt(GAMMA * R_AIR * t_inf)
    return 0.5 * GAMMA * p_inf * ma_inf**2, rho_inf, v_inf


def load_ref(path):
    x, y = np.loadtxt(path, comments="#", unpack=True)
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def peak(x, y):
    i = int(np.argmax(y))
    return float(x[i]), float(y[i])


def onset(x, y, thr=0.05):
    hit = np.where(y > thr)[0]
    return float(x[hit[0]]) if hit.size else float("nan")


def plot_clcd(case, outdir, ifcl=True, ifcd=True):
    h = read_history(Path(case) / "history.csv")
    m = h["phase"] == "U"
    if not m.any():
        m = np.ones(h["time"].size, bool)
    t, cl, cd = h["time"][m], h["Cl"][m], h["Cd"][m]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    if ifcd:
        ax.plot(t, cd, color=C_LOWER, lw=1.1, label=r"$C_d$")
    if ifcl:
        ax.plot(t, cl, color=C_UPPER, lw=1.1, label=r"$C_l$")
    ax.set_xlabel(r"$t$ / s")
    ax.set_ylabel("coefficient")
    ax.set_title("Aerodynamic coefficients")
    ax.grid(alpha=0.3, lw=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = Path(outdir) / "clcd.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def phase_analysis(case, outdir, nbins, tmin, uminf, fmin, fmax):
    case = Path(case)
    outdir = Path(outdir)

    hist = read_history(case / "history.csv")
    t_probe, cp_probe = read_probe(case / "cp_probe_x0.45.dat")
    t_wall, wall = read_wall_cp(case / "wall_cp.dat")
    xw, yw = wall[0, :, 0], wall[0, :, 1]
    cp_wall = wall[:, :, 2]

    upper, lower, chord = split_sides(xw, yw)
    xc = xw / chord
    idx_up = np.where(upper)[0][np.argsort(xc[upper])]
    idx_lo = np.where(lower)[0][np.argsort(xc[lower])]

    th = hist["time"]
    mh = th >= tmin
    f_cl, coef_cl, r2_cl, spec_cl = spectral_peak(th[mh], hist["Cl"][mh], fmin, fmax)
    f_cd, coef_cd, r2_cd, spec_cd = spectral_peak(th[mh], hist["Cd"][mh], fmin, fmax)

    mp = t_probe >= tmin
    f_pr, coef_pr, r2_pr, spec_pr = spectral_peak(t_probe[mp], cp_probe[mp], fmin, fmax)

    mw = t_wall >= tmin
    f_nodes = []
    for j in range(cp_wall.shape[1]):
        fj, aj = amplitude_spectrum(t_wall[mw], cp_wall[mw, j])
        sel = (fj >= fmin) & (fj <= fmax)
        f_nodes.append(fj[sel][np.argmax(aj[sel])])
    f_nodes = np.asarray(f_nodes)
    f0 = float(np.median(f_nodes))
    period = 1.0 / f0
    st = f0 * chord / uminf

    a_cl = np.column_stack(
        [np.ones(mh.sum()), np.cos(2 * np.pi * f0 * th[mh]), np.sin(2 * np.pi * f0 * th[mh])]
    )
    coef_cl_f0 = np.linalg.lstsq(a_cl, hist["Cl"][mh], rcond=None)[0]
    t_peak = peak_time(coef_cl_f0, f0)

    centers, cpb_all, counts = phase_average(cp_wall[mw], t_wall[mw], f0, t_peak, nbins)
    targets = np.arange(nbins) / nbins
    snap_all = periodic_eval(centers, cpb_all, targets)
    mean_cp = snap_all.mean(axis=0)
    rms_cp = np.sqrt(np.mean((snap_all - mean_cp) ** 2, axis=0))
    j_probe = idx_up[np.argmin(np.abs(xc[idx_up] - 0.45))]

    for name, idx in (("upper", idx_up), ("lower", idx_lo)):
        np.savetxt(
            outdir / f"cp_phase_mean_{name}.dat",
            np.column_stack([xc[idx], mean_cp[idx], rms_cp[idx]]),
            fmt="%.8e",
            header=f"x/c  Cp_mean_{name}  Cp_rms_{name}   (period-averaged, f0={f0:.4f} Hz)",
        )
        np.savetxt(
            outdir / f"cp_phase_{nbins:02d}_{name}.dat",
            np.column_stack([xc[idx]] + [snap_all[k, idx] for k in range(nbins)]),
            fmt="%.8e",
            header="x/c  " + "  ".join(f"Cp(phi={k}/{nbins})" for k in range(nbins)),
        )

    summary = {
        "case": str(case),
        "chord_m": chord,
        "u_inf_m_s": uminf,
        "f0_hz": f0,
        "period_s": period,
        "strouhal": st,
        "f_cl_hz": f_cl,
        "f_cd_hz": f_cd,
        "f_probe_hz": f_pr,
        "f_nodes_median_hz": float(np.median(f_nodes)),
        "f_nodes_p10_p90_hz": [
            float(np.percentile(f_nodes, 10)),
            float(np.percentile(f_nodes, 90)),
        ],
        "Cl_mean": float(hist["Cl"][mh].mean()),
        "Cl_amplitude_fundamental": float(np.hypot(coef_cl_f0[1], coef_cl_f0[2])),
        "R2_Cl_fundamental": float(r2_cl),
        "probe_x_over_c": float(xc[j_probe]),
        "probe_phase_mean_range": [
            float(mean_cp[j_probe] - rms_cp[j_probe]),
            float(mean_cp[j_probe] + rms_cp[j_probe]),
        ],
        "phase_bin_counts": counts.tolist(),
        "tmin": tmin,
        "nbins": nbins,
    }
    (outdir / "fft_peaks.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4))

    ax = axes[0, 0]
    ax.plot(th[mh], hist["Cl"][mh], lw=1.0, color="#333333")
    ax.set_xlabel(r"$t$ / s")
    ax.set_ylabel(r"$C_l$")
    ax.set_title(f"(a) $C_l(t)$,  $T$ = {period * 1e3:.2f} ms,  $f_0$ = {f0:.2f} Hz")
    ax.grid(alpha=0.3, lw=0.6)

    ax = axes[0, 1]
    for spec, color, label in (
        (spec_cl, C_UPPER, r"$C_l$"),
        (spec_cd, C_LOWER, r"$C_d$"),
        (spec_pr, "#e67e22", r"$C_p$ probe (x/c=0.45)"),
    ):
        sel = (spec[0] >= fmin) & (spec[0] <= fmax)
        ax.semilogy(spec[0][sel], spec[1][sel], lw=1.0, color=color, label=label)
    ax.axvline(f0, color="k", ls="--", lw=0.9)
    ax.set_xlabel(r"$f$ / Hz")
    ax.set_ylabel("amplitude")
    ax.set_title(f"(b) amplitude spectra,  St = {st:.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.6)

    ax = axes[1, 0]
    phi_p = np.mod((t_probe[mp] - t_peak) * f0, 1.0)
    ax.plot(phi_p, cp_probe[mp], ".", ms=3, color="#888888", alpha=0.6, label="samples")
    ctr, cpb_probe, _ = phase_average(cp_probe[mp, None], t_probe[mp], f0, t_peak, nbins)
    grid = np.linspace(0, 1, 400)
    fold = periodic_eval(ctr, cpb_probe, grid)
    ax.plot(grid, fold[:, 0], "-", color=C_UPPER, lw=1.6, label=f"phase average ({nbins} bins)")
    ax.set_xlabel(r"$\phi = (t-t_{C_l,\max})/T$")
    ax.set_ylabel(r"$C_p$ at $x/c=0.45$")
    ax.set_title("(c) phase folding of probe signal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.6)

    ax = axes[1, 1]
    fluc = cp_wall[mw] - cp_wall[mw].mean(axis=0)
    amp_node = 2.0 * np.abs(np.exp(-1j * 2 * np.pi * f0 * t_wall[mw]) @ fluc) / mw.sum()
    ax.plot(xc[idx_lo], amp_node[idx_lo], color=C_LOWER, lw=1.3, label="lower")
    ax.plot(xc[idx_up], amp_node[idx_up], color=C_UPPER, lw=1.3, label="upper")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$|\hat{C}_p|$ at $f_0$")
    ax.set_title(f"(d) fundamental amplitude, node peak $f$: median {np.median(f_nodes):.2f} Hz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "fft_spectrum.png")
    plt.close(fig)

    xu, xl = xc[idx_up], xc[idx_lo]
    mu, ml = mean_cp[idx_up], mean_cp[idx_lo]
    ru, rl = rms_cp[idx_up], rms_cp[idx_lo]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.fill_between(xu, mu - ru, mu + ru, color=C_UPPER, alpha=0.18, lw=0)
    ax.plot(xu, mu, color=C_UPPER, lw=1.8, label="period mean, upper")
    ax.fill_between(xl, ml - rl, ml + rl, color=C_LOWER, alpha=0.18, lw=0)
    ax.plot(xl, ml, color=C_LOWER, lw=1.8, label="period mean, lower")

    ref_file = case / "cp_mean.dat"
    ref = read_cp_mean(ref_file) if ref_file.exists() else {}
    if "upper" in ref:
        ax.plot(
            ref["upper"][:, 0], ref["upper"][:, 1], "--", color=C_NEUTRAL, lw=1.0,
            label="reference time mean",
        )
    if "lower" in ref:
        ax.plot(ref["lower"][:, 0], ref["lower"][:, 1], "--", color=C_NEUTRAL, lw=1.0)

    ax.invert_yaxis()
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$C_p$")
    ax.set_title(
        f"Single-period mean $C_p$  ($f_0$={f0:.2f} Hz, $T$={period * 1e3:.2f} ms; "
        r"band = $\pm$1 r.m.s.)"
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3, lw=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "cp_period_mean.png")
    plt.close(fig)

    ncol = 4
    nrow = int(np.ceil(nbins / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 2.1 * nrow + 0.6), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    ymax = float(np.nanmax(snap_all[:, idx_up]))
    ymin = float(np.nanmin(snap_all[:, idx_up]))
    for k in range(nbins):
        ax = axes[k // ncol, k % ncol]
        ax.plot(xu, snap_all[k, idx_up], color=C_UPPER, lw=1.3)
        ax.plot(xl, snap_all[k, idx_lo], color=C_LOWER, lw=1.3)
        ax.set_title(rf"$\phi = {k}/{nbins}$  ($t/T={k / nbins:.3f}$)", fontsize=9)
        ax.grid(alpha=0.28, lw=0.5)
        if k % ncol == 0:
            ax.set_ylabel(r"$C_p$")
        if k // ncol == nrow - 1:
            ax.set_xlabel(r"$x/c$")
    for k in range(nbins, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    axes[0, 0].set_ylim(ymax + 0.15, ymin - 0.15)
    axes[0, 0].plot([], [], color=C_UPPER, label="upper")
    axes[0, 0].plot([], [], color=C_LOWER, label="lower")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"$C_p$ distribution every $T/{nbins}$  ($\\phi=0$ at $C_l^{{\\max}}$)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(outdir / f"cp_phase_{nbins:02d}.png")
    plt.close(fig)

    cmap = plt.get_cmap("twilight_shifted")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for k in range(nbins):
        col = cmap(k / nbins)
        axes[0].plot(xu, snap_all[k, idx_up], color=col, lw=1.0)
        axes[1].plot(xl, snap_all[k, idx_lo], color=col, lw=1.0)
    for ax, name in zip(axes, ("upper", "lower")):
        ax.set_xlabel(r"$x/c$")
        ax.set_title(f"{name} surface")
        ax.grid(alpha=0.3, lw=0.6)
    axes[0].set_ylabel(r"$C_p$")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.03)
    cb.set_label(r"$\phi = t/T$")
    fig.suptitle(f"$C_p$ at $\\phi = 0, 1/{nbins}, \\ldots, {nbins - 1}/{nbins}$", fontsize=11)
    fig.savefig(outdir / f"cp_phase_{nbins:02d}_overlay.png")
    plt.close(fig)

    print(f"case           : {case}")
    print(f"chord c        : {chord:.4f} m      U_inf = {uminf:.1f} m/s")
    print(f"wall blocks    : {len(t_wall)}  ({t_wall[0]:.4e} .. {t_wall[-1]:.4e} s), nodes = {xw.size}")
    nper = (t_wall[mw][-1] - t_wall[mw][0] + np.median(np.diff(t_wall))) * f0
    print(f"analysis window: t >= {tmin} s  ({mw.sum()} blocks, {nper:.2f} periods)")
    print(f"f0 (wall nodes): {f0:.3f} Hz   T = {period * 1e3:.3f} ms   St = {st:.4f}")
    print(f"f(Cl) = {f_cl:.3f} Hz | f(Cd) = {f_cd:.3f} Hz | f(probe) = {f_pr:.3f} Hz")
    print(
        f"node f0 spread : p10..p90 = {np.percentile(f_nodes, 10):.2f} .. "
        f"{np.percentile(f_nodes, 90):.2f} Hz"
    )
    print(
        f"Cl mean = {summary['Cl_mean']:.4f}, "
        f"fundamental amplitude = {summary['Cl_amplitude_fundamental']:.4f}"
    )
    print(f"phase-zero t   : {t_peak:.6f} s  (Cl maximum)")
    print(
        f"probe x/c      : {xc[j_probe]:.4f}, period mean Cp = {mean_cp[j_probe]:+.4f} "
        f"(rms {rms_cp[j_probe]:.4f})"
    )
    print(f"bins           : {nbins}, samples per bin = {counts.min()} .. {counts.max()}")


def plot_cp_rms(outdir, refs, q, focus):
    outdir = Path(outdir)
    ours = np.loadtxt(outdir / "cp_phase_mean_upper.dat")
    x_our, rms_our = ours[:, 0], ours[:, 2]

    curves = []
    for label, path, color, marker, scale in refs:
        path = Path(path)
        if not path.exists():
            continue
        xr, yr = load_ref(path)
        curves.append({"label": label, "x": xr, "y": yr * scale, "color": color, "marker": marker})

    clip_file = outdir / "cp_mean_rms_raw_vs_clip_upper.dat"
    have_clip = clip_file.exists()
    if have_clip:
        cl = np.loadtxt(clip_file)
        x_cl, rms_cl = cl[:, 0], cl[:, 4]

    if curves:
        fig, (ax, axd) = plt.subplots(
            2, 1, figsize=(8.6, 5.8), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0]}
        )
    else:
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        axd = None

    for r in curves:
        xp, yp = peak(r["x"], r["y"])
        ax.plot(
            r["x"],
            r["y"],
            "-",
            marker=r["marker"],
            ms=3.4,
            mfc="white",
            mec=r["color"],
            mew=1.0,
            lw=1.1,
            color=r["color"],
            zorder=2,
            label=f"{r['label']} (peak {yp:.3f} @ {xp:.3f})",
        )
    xp, yp = peak(x_our, rms_our)
    ax.plot(
        x_our,
        rms_our,
        "-",
        color=C_OURS,
        lw=2.0,
        zorder=3,
        label=f"present, period r.m.s. (peak {yp:.3f} @ {xp:.3f})",
    )
    if have_clip:
        ax.plot(x_cl, rms_cl, "--", color=C_CLIP, lw=1.2, zorder=1, label="present, overshoot clipped")

    ax.set_ylabel(r"$C_{p,\mathrm{rms}}$")
    ax.set_title(r"Wall-pressure r.m.s. along the chord (upper surface)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, max([float(rms_our.max()), *(float(r["y"].max()) for r in curves)]) * 1.15)
    ax.grid(alpha=0.3, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.2)

    if axd is not None:
        axd.axhline(0, color="k", lw=0.8)
        for r in curves:
            m = (r["x"] >= focus[0]) & (r["x"] <= focus[1]) & (r["x"] <= x_our.max())
            diff = np.interp(r["x"][m], x_our, rms_our) - r["y"][m]
            axd.plot(
                r["x"][m],
                diff,
                "-",
                color=r["color"],
                lw=1.4,
                label=f"present $-$ {r['label']}  (mean |$\\Delta$| = {np.abs(diff).mean():.3f})",
            )
        axd.set_xlabel(r"$x/c$")
        axd.set_ylabel(r"$\Delta C_{p,\mathrm{rms}}$")
        axd.grid(alpha=0.3, lw=0.6)
        axd.legend(loc="upper left", fontsize=8)
    else:
        ax.set_xlabel(r"$x/c$")

    fig.tight_layout()
    fig.savefig(outdir / "cp_rms_compare.png")
    plt.close(fig)

    print(f"present peak    : {yp:.4f} at x/c = {xp:.4f}   onset(0.05) = {onset(x_our, rms_our):.3f}")
    for r in curves:
        xp_r, yp_r = peak(r["x"], r["y"])
        m = (r["x"] >= focus[0]) & (r["x"] <= focus[1]) & (r["x"] <= x_our.max())
        diff = np.interp(r["x"][m], x_our, rms_our) - r["y"][m]
        print(
            f"{r['label']:<22s}: peak {yp_r:.4f} at x/c = {xp_r:.4f}   "
            f"onset = {onset(r['x'], r['y']):.3f}   "
            f"mean |diff| = {np.abs(diff).mean():.4f}, max |diff| = {np.abs(diff).max():.4f} "
            f"@ {r['x'][m][np.argmax(np.abs(diff))]:.3f}"
        )
    if np.isfinite(q):
        print(f"dynamic pressure: q = {q:.1f} Pa")


def main():
    ap = argparse.ArgumentParser(description="OAT15A post-processing")
    ap.add_argument("--case", default=str(DEFAULT_CASE))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--nbins", type=int, default=16)
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--uminf", type=float, default=253.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=400.0)
    ap.add_argument("--ref-sa", default=str(REPO_ROOT / "refs" / "oat15a_cp_rms_sa_ref.dat"))
    ap.add_argument("--ref-exp", default=str(REPO_ROOT / "refs" / "oat15a_cp_rms_exp_ref.dat"))
    ap.add_argument("--ref-des", default=str(REPO_ROOT / "refs" / "oat15a_p_rms_zonal_des.dat"))
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--focus", type=float, nargs=2, default=(0.0, 1.0))
    ap.add_argument("--no-clcd", action="store_true")
    ap.add_argument("--no-spectrum", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    case = Path(args.case)
    outdir = Path(args.out) if args.out else default_outdir(case, args.data_root)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.no_clcd:
        if (case / "history.csv").exists():
            print(f"clcd            : {plot_clcd(case, outdir)}")
        else:
            print(f"clcd            : skipped (no {case / 'history.csv'})")

    if not args.no_spectrum:
        need = [case / "history.csv", case / "cp_probe_x0.45.dat", case / "wall_cp.dat"]
        if all(p.exists() for p in need):
            phase_analysis(case, outdir, args.nbins, args.tmin, args.uminf, args.fmin, args.fmax)
        else:
            print("spectrum        : skipped (missing history / cp_probe / wall_cp)")

    if not args.no_compare:
        if (outdir / "cp_phase_mean_upper.dat").exists():
            des_ok = Path(args.ref_des).exists()
            q = args.q
            if des_ok and q is None:
                q, rho_inf, v_inf = farfield_q(case)
                print(
                    f"far field       : rho_inf = {rho_inf:.4f} kg/m3, V_inf = {v_inf:.2f} m/s"
                )
            scale_des = 1.0 / q if des_ok else 1.0
            refs = [
                ("SA ref", args.ref_sa, C_SA, "o", 1.0),
                ("Exp ref", args.ref_exp, C_EXP, "s", 1.0),
                (r"Zonal DES ($p_{rms}/q$)", args.ref_des, C_DES, "^", scale_des),
            ]
            plot_cp_rms(outdir, refs, q if q is not None else float("nan"), args.focus)
        else:
            print("compare         : skipped (no cp_phase_mean_upper.dat)")

    print(f"outputs         : {outdir}")


if __name__ == "__main__":
    main()