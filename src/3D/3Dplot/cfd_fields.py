"""流场云图：中展向截面 u/v、y=0 俯视 w 与宽视角涡结构。"""

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from cfd_style import CMAP_FLOW, save_show

SECTION_XLIM = (-2.0, 5.0)
SECTION_YLIM = (-2.0, 2.0)
SECTION_PAD = 0.45
WIDE_XLIM = (-3.0, 12.0)
WIDE_YLIM = (-3.0, 3.0)
WIDE_PAD = 0.6


@dataclass
class FieldData:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    r: np.ndarray
    rho_inf: float
    p_inf: float
    u_inf: float
    q_inf: float
    z_planes: np.ndarray
    n_r: int
    n_t: int
    z_mid: float
    z_sec: tuple
    x_sec: np.ndarray
    y_sec: np.ndarray
    u_sec: np.ndarray
    v_sec: np.ndarray


def load_field(case):
    field = pd.read_csv(case.case_dir / 'field.csv')
    x = field['x'].to_numpy()
    y = field['y'].to_numpy()
    z = field['z'].to_numpy()
    u = field['u'].to_numpy()
    v = field['v'].to_numpy()
    w = field['w'].to_numpy()
    r = np.hypot(x, y)

    far = r > 0.7 * r.max()
    rho_inf = float(np.median(field['rho'].to_numpy()[far]))
    p_inf = float(np.median(field['p'].to_numpy()[far]))
    u_inf = float(np.median(u[far]))
    q_inf = 0.5 * rho_inf * u_inf ** 2

    z_planes = np.unique(z)
    first = np.isclose(z, z_planes[0])
    n_t = int(np.unique(np.round(np.degrees(np.arctan2(y[first], x[first])), 4)).size)
    n_r = int(first.sum() // n_t)

    z_mid = 0.5 * (z_planes.min() + z_planes.max())
    pair = np.argsort(np.abs(z_planes - z_mid))[:2]
    low = np.isclose(z, z_planes[pair[0]])
    high = np.isclose(z, z_planes[pair[1]])

    data = FieldData(
        x=x, y=y, z=z, u=u, v=v, w=w, r=r,
        rho_inf=rho_inf, p_inf=p_inf, u_inf=u_inf, q_inf=q_inf,
        z_planes=z_planes, n_r=n_r, n_t=n_t, z_mid=z_mid,
        z_sec=(float(z_planes[pair[0]]), float(z_planes[pair[1]])),
        x_sec=x[low], y_sec=y[low],
        u_sec=0.5 * (u[low] + u[high]), v_sec=0.5 * (v[low] + v[high]),
    )

    print(f'field nodes = {x.size}')
    print(f'planes = {z_planes.size}, radial = {n_r}, tangential = {n_t}')
    print(f'z/D = {z_planes.min():.3f} ... {z_planes.max():.3f}, mid-span = {z_mid:.3f}')
    print(f'section from z/D = {data.z_sec[0]:.3f} and {data.z_sec[1]:.3f}')
    print(f'D = {case.diam:.4f} m, rho_inf = {rho_inf:.4e} kg/m3, U_inf = {u_inf:.3f} m/s')
    print(f'p_inf = {p_inf:.4f} Pa, q_inf = {q_inf:.6f} Pa')

    params_file = case.case_dir / 'parameters.json'
    if params_file.exists():
        params = json.loads(params_file.read_text(encoding='utf-8'))
        print(
            f"parameters.json: Ma = {params['Ma']}, Re = {params['Re']}, "
            f"span = {params['span_length_D']} D, span boundary = {params['span_boundary']}"
        )

    return data


def _triangulation(xs, ys, radius, xlim, ylim, pad):
    keep = (
        (xs > xlim[0] - pad)
        & (xs < xlim[1] + pad)
        & (ys > ylim[0] - pad)
        & (ys < ylim[1] + pad)
    )
    tri = mtri.Triangulation(xs[keep], ys[keep])
    centers = np.column_stack(
        (xs[keep][tri.triangles].mean(axis=1), ys[keep][tri.triangles].mean(axis=1))
    )
    tri.set_mask(np.hypot(centers[:, 0], centers[:, 1]) < radius)
    return tri, keep


def _cylinder_patch(radius):
    return plt.Circle((0.0, 0.0), radius, facecolor='white', edgecolor='k', linewidth=0.9, zorder=3)


def _plane_figure(tri, values, label, title, xlim, ylim, radius, figsize, n_levels=22, symmetric=False):
    if symmetric:
        vmax = float(np.abs(values).max())
        levels = np.linspace(-vmax, vmax, n_levels)
    else:
        levels = np.linspace(values.min(), values.max(), n_levels)

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    filled = ax.tricontourf(tri, values, levels=levels, cmap=CMAP_FLOW, extend='both')
    ax.tricontour(tri, values, levels=levels[::3], colors='k', linewidths=0.3, alpha=0.5)
    ax.add_patch(_cylinder_patch(radius))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x/D')
    ax.set_ylabel('y/D')
    ax.set_title(title)

    bar = fig.colorbar(filled, ax=ax, orientation='horizontal', pad=0.13, extend='both')
    bar.set_ticks(np.linspace(levels[0], levels[-1], 7))
    bar.ax.set_title(label)
    return fig


def plot_sections(case, field):
    tri, keep = _triangulation(
        field.x_sec, field.y_sec, case.radius, SECTION_XLIM, SECTION_YLIM, SECTION_PAD
    )
    for values, label, name, sym in (
        (field.u_sec, r'$u$ [m/s]', 'u_section.png', False),
        (field.v_sec, r'$v$ [m/s]', 'v_section.png', True),
    ):
        local = values[keep]
        fig = _plane_figure(
            tri, local, label, f'{label}, z/D = {field.z_mid:.3f}',
            SECTION_XLIM, SECTION_YLIM, case.radius, (6.6, 4.0), symmetric=sym,
        )
        save_show(fig, case.fig_dir, name)
        print(f'{label}: {local.min():.3f} ... {local.max():.3f}')

    print(f'section nodes in view = {int(keep.sum())}, masked triangles = {int(tri.mask.sum())}')
    return tri, keep


def plot_w_top(case, field):
    rows = []
    r_plane = None
    for plane in field.z_planes:
        on_plane = np.isclose(field.z, plane)
        w_plane = field.w[on_plane].reshape(field.n_r, field.n_t)
        r_plane = field.r[on_plane].reshape(field.n_r, field.n_t)[:, 0]
        w_front = 0.5 * (w_plane[:, 0] + w_plane[:, -1])
        w_back = 0.5 * (w_plane[:, field.n_t // 2 - 1] + w_plane[:, field.n_t // 2])
        rows.append(np.concatenate((w_back[::-1], w_front)))

    w_slice = np.asarray(rows)
    x_slice = np.concatenate((-r_plane[::-1], r_plane))

    x_fine = np.linspace(SECTION_XLIM[0], SECTION_XLIM[1], 401)
    z_fine = np.linspace(field.z_planes.min(), field.z_planes.max(), 241)
    zz_fine, xx_fine = np.meshgrid(z_fine, x_fine, indexing='ij')
    w_fine = RegularGridInterpolator((field.z_planes, x_slice), w_slice)(
        np.column_stack((zz_fine.ravel(), xx_fine.ravel()))
    ).reshape(zz_fine.shape)

    w_max = float(np.abs(w_fine).max())
    levels = np.linspace(-w_max, w_max, 22)
    fig_h = float(
        np.clip(2.4 + 5.0 * np.ptp(field.z_planes) / (SECTION_XLIM[1] - SECTION_XLIM[0]), 2.6, 5.4)
    )

    fig, ax = plt.subplots(figsize=(6.8, fig_h), layout='constrained')
    filled = ax.contourf(x_fine, z_fine, w_fine, levels=levels, cmap=CMAP_FLOW, extend='both')
    ax.contour(x_fine, z_fine, w_fine, levels=levels[::3], colors='k', linewidths=0.3, alpha=0.5)
    ax.add_patch(
        plt.Rectangle(
            (-case.radius, field.z_planes.min()),
            case.diam,
            field.z_planes.max() - field.z_planes.min(),
            facecolor='white',
            edgecolor='k',
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.set_xlim(*SECTION_XLIM)
    ax.set_ylim(field.z_planes.min(), field.z_planes.max())
    ax.set_xlabel('x/D')
    ax.set_ylabel('z/D')
    ax.set_title(r'$w$ on $y/D$ = 0')
    bar = fig.colorbar(filled, ax=ax, orientation='horizontal', pad=0.13, extend='both')
    bar.set_ticks(np.linspace(levels[0], levels[-1], 7))
    bar.ax.set_title(r'$w$ [m/s]')

    print(f'slice points = {w_slice.size}, x = {x_slice.min():.3f} ... {x_slice.max():.3f}')
    print(f'spanwise w rms = {field.w.std():.4f} m/s, |w|max = {w_max:.4f} m/s')

    return save_show(fig, case.fig_dir, 'w_top.png')


def plot_wide_view(case, field):
    tri, keep = _triangulation(
        field.x_sec, field.y_sec, case.radius, WIDE_XLIM, WIDE_YLIM, WIDE_PAD
    )
    span = WIDE_XLIM[1] - WIDE_XLIM[0]
    height = WIDE_YLIM[1] - WIDE_YLIM[0]
    panel_h = 8.6 * height / span
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 2.0 * panel_h + 2.4), layout='constrained')

    for ax, values, label, sym in (
        (axes[0], field.u_sec[keep], r'$u$ [m/s]', False),
        (axes[1], field.v_sec[keep], r'$v$ [m/s]', True),
    ):
        if sym:
            vmax = float(np.abs(values).max())
            levels = np.linspace(-vmax, vmax, 26)
        else:
            levels = np.linspace(values.min(), values.max(), 26)
        filled = ax.tricontourf(tri, values, levels=levels, cmap=CMAP_FLOW, extend='both')
        ax.tricontour(tri, values, levels=levels[::4], colors='k', linewidths=0.3, alpha=0.5)
        ax.add_patch(_cylinder_patch(case.radius))
        ax.set_xlim(*WIDE_XLIM)
        ax.set_ylim(*WIDE_YLIM)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x/D')
        ax.set_ylabel('y/D')
        ax.set_title(f'{label}, z/D = {field.z_mid:.3f}, wide view')
        bar = fig.colorbar(filled, ax=ax, orientation='horizontal', pad=0.02, extend='both')
        bar.set_ticks(np.linspace(levels[0], levels[-1], 7))
        bar.ax.set_title(label)

    print(f'wide view: x/D = {WIDE_XLIM[0]} ... {WIDE_XLIM[1]}, y/D = {WIDE_YLIM[0]} ... {WIDE_YLIM[1]}')
    print(f'nodes in view = {int(keep.sum())}')
    print(f'u = {field.u_sec[keep].min():.3f} ... {field.u_sec[keep].max():.3f} m/s')
    print(f'v = {field.v_sec[keep].min():.3f} ... {field.v_sec[keep].max():.3f} m/s')

    return save_show(fig, case.fig_dir, 'wide_view.png')
