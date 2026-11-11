"""Q 准则三维涡识别：柱坐标结构化网格上的 Q 场、涡量与等值面。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import label as ndi_label
from scipy.ndimage import map_coordinates
from skimage import measure

from cfd_style import save_show

Q_R_MAX = 4.0
Q_R_MIN_FACTOR = 1.10
Q_LEVEL = 2.0
Q_VIEW = (30.0, 130.0)
Q_ZOOM = 1.70
Q_XLIM = (-1.3, 3.2)
Q_YLIM = (-1.5, 1.5)


def _dtheta(phi, dtheta):
    return (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * dtheta)


def compute_vorticity(case, field):
    """返回 (Q, omega_z, |omega|)，无量纲化：Q D^2/U^2 与 omega D/U。"""
    nz, nr, nt = field.z_planes.size, field.n_r, field.n_t
    x = field.x.reshape(nz, nr, nt)
    y = field.y.reshape(nz, nr, nt)
    r = field.r.reshape(nz, nr, nt)
    ct, st = x / r, y / r

    u = field.u.reshape(nz, nr, nt)
    v = field.v.reshape(nz, nr, nt)
    w = field.w.reshape(nz, nr, nt)

    r_axis = r[0, :, 0]
    z_axis = field.z_planes
    dtheta = 2.0 * np.pi / nt

    def cart_grad(phi):
        d_r = np.gradient(phi, r_axis, axis=1)
        d_t = _dtheta(phi, dtheta)
        d_z = np.gradient(phi, z_axis, axis=0)
        return d_r * ct - d_t * st / r, d_r * st + d_t * ct / r, d_z

    u_x, u_y, u_z = cart_grad(u)
    v_x, v_y, v_z = cart_grad(v)
    w_x, w_y, w_z = cart_grad(w)

    strain = (
        u_x ** 2 + v_y ** 2 + w_z ** 2
        + 0.5 * (u_y + v_x) ** 2
        + 0.5 * (u_z + w_x) ** 2
        + 0.5 * (v_z + w_y) ** 2
    )
    spin = (
        0.5 * (u_y - v_x) ** 2
        + 0.5 * (u_z - w_x) ** 2
        + 0.5 * (v_z - w_y) ** 2
    )
    q = 0.5 * (spin - strain)

    omega_x = w_y - v_z
    omega_y = u_z - w_x
    omega_z = v_x - u_y

    scale_q = (case.diam / field.u_inf) ** 2
    scale_w = case.diam / field.u_inf
    omega_mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)
    return q * scale_q, omega_z * scale_w, omega_mag * scale_w


def _cylindrical_vertices(verts, r_axis, z_axis, n_t):
    rr = np.interp(verts[:, 1], np.arange(r_axis.size), r_axis)
    zz = np.interp(verts[:, 0], np.arange(z_axis.size), z_axis)
    tt = verts[:, 2] * (2.0 * np.pi / n_t)
    return rr * np.cos(tt), rr * np.sin(tt), zz


def _cylinder_outline(ax, radius, z_lo, z_hi, color="0.35", lw=1.4):
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    cx, cy = radius * np.cos(theta), radius * np.sin(theta)
    th_grid, z_grid = np.meshgrid(theta, np.linspace(z_lo, z_hi, 2))
    ax.plot_surface(
        radius * np.cos(th_grid),
        z_grid,
        radius * np.sin(th_grid),
        color="0.55",
        alpha=0.30,
        linewidth=0,
        shade=False,
        zorder=5,
    )
    for z_end in (z_lo, z_hi):
        ax.plot(cx, np.full_like(cx, z_end), cy, color=color, lw=lw, alpha=0.55, zorder=30)


def _refine(arr, shape_out):
    nz, nr, ntc = arr.shape
    grid = np.meshgrid(
        np.linspace(0.0, nz - 1.0, shape_out[0]),
        np.linspace(0.0, nr - 1.0, shape_out[1]),
        np.linspace(0.0, ntc - 1.0, shape_out[2]),
        indexing="ij",
    )
    return map_coordinates(arr, np.stack(grid), order=3, mode="nearest")


def plot_q_criterion(
    case,
    field,
    level=Q_LEVEL,
    r_max=Q_R_MAX,
    r_min_factor=Q_R_MIN_FACTOR,
    view=Q_VIEW,
    xlim=Q_XLIM,
    ylim=Q_YLIM,
    zoom=Q_ZOOM,
):
    q, _omega_z, omega_mag = compute_vorticity(case, field)
    nz, nr, nt = q.shape
    r_axis = field.r.reshape(nz, nr, nt)[0, :, 0]
    keep_r = (r_axis <= r_max) & (r_axis >= case.radius * r_min_factor)
    if keep_r.sum() < 5:
        print(f"no radial points in {r_min_factor:.2f}R <= r <= {r_max}D, Q criterion skipped")
        return None

    def extend(array):
        band = array[:, keep_r, :]
        return np.concatenate((band, band[:, :, :1]), axis=2)

    q_ext = extend(q)
    om_ext = extend(omega_mag)
    q_max = float(q_ext.max())
    if not np.isfinite(q_max) or level >= q_max:
        print(f"Q level {level:g} exceeds field maximum {q_max:.3f}, iso-surface skipped")
        return None

    ref = (q_ext.shape[0] * 2, q_ext.shape[1] * 2, q_ext.shape[2] * 2)
    q_ref = _refine(q_ext, ref)
    om_ref = _refine(om_ext, ref)
    r_ref = np.interp(
        np.linspace(0.0, r_axis[keep_r].size - 1.0, ref[1]), np.arange(r_axis[keep_r].size), r_axis[keep_r]
    )
    n_t_ref = ref[2] - 1

    verts, faces, normals, _ = measure.marching_cubes(q_ref, level=level, spacing=(1.0, 1.0, 1.0))
    vx, vy, vz = _cylindrical_vertices(verts, r_ref, field.z_planes, n_t_ref)
    om_vertex = map_coordinates(om_ref, verts.T, order=3)
    om_face = om_vertex[faces].mean(axis=1)
    triangles = np.stack([np.column_stack([vx[f], vz[f], vy[f]]) for f in faces])

    light = np.array([0.40, -0.35, 0.85])
    light /= np.linalg.norm(light)
    shade_vertex = 0.74 + 0.34 * np.abs(normals @ light)

    vmin = float(np.percentile(om_face, 2.0))
    vmax = float(np.percentile(om_face, 98.0))
    norm = colors.PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("turbo")
    col_vertex = cmap(norm(om_vertex))[:, :3] * shade_vertex[:, None]
    face_rgb = np.clip(col_vertex[faces].mean(axis=1), 0.0, 1.0)

    _lab, n_blobs = ndi_label(np.pad(q_ext > level, 1))

    z_lo, z_hi = float(field.z_planes.min()), float(field.z_planes.max())
    fig = plt.figure(figsize=(10.0, 7.2))
    ax = fig.add_axes([0.0, 0.0, 0.94, 0.96], projection="3d")
    ax.patch.set_alpha(0.0)

    ax.add_collection3d(
        Poly3DCollection(triangles, facecolors=face_rgb, linewidths=0.0, antialiased=False)
    )
    _cylinder_outline(ax, case.radius, z_lo, z_hi)

    ax.set_xlim(*xlim)
    ax.set_ylim(z_lo, z_hi)
    ax.set_zlim(*ylim)
    ax.set_box_aspect((xlim[1] - xlim[0], z_hi - z_lo, ylim[1] - ylim[0]), zoom=zoom)
    ax.set_axis_off()
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(
        f"Q iso-surface $QD^2/U_\\infty^2$ = {level:g} colored by $|\\omega| D/U_\\infty$", pad=-4
    )

    cax = fig.add_axes([0.945, 0.30, 0.014, 0.36])
    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.ax.set_title(r"$|\omega| D/U_\infty$", fontsize=8, pad=6)

    print(f"Q over {r_min_factor:.2f}R <= r <= {r_max}D: {q_ext.min():.2f} ... {q_ext.max():.2f}")
    print(f"Q iso-surface level = {level:g}, triangles = {faces.shape[0]}, connected regions = {n_blobs}")
    print(f"face |omega| D/U_inf = {om_face.min():.2f} ... {om_face.max():.2f}, color range {vmin:.2f} ... {vmax:.2f}")
    print("note: Q is almost uniform along z (spanwise std 0.06 vs radial 2.49), so vortices appear as spanwise rollers")

    return save_show(fig, case.fig_dir, "q_criterion.png")
