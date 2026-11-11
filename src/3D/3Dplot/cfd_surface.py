"""圆柱表面压力系数：展向分布与时变/时均曲线。"""

import matplotlib.pyplot as plt
import numpy as np

from cfd_style import C_GRAY, C_GUIDE, C_GRID, C_RED, save_show


def _sorted_surface(case):
    order = np.argsort(case.theta_wall)
    return case.theta_wall[order], case.cp_wall[:, order]


def plot_cp_surface(case):
    theta_axis, cp_sorted = _sorted_surface(case)

    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout='constrained')
    for k in range(case.z_wall.size):
        ax.plot(theta_axis, cp_sorted[k], color=C_GRAY, linewidth=0.5)
    ax.plot(theta_axis, cp_sorted.mean(axis=0), color=C_RED, linewidth=1.7)
    ax.axhline(0.0, color=C_GUIDE, linewidth=0.7, linestyle='--')
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'$C_p$')
    ax.set_title(
        f'surface $C_p$, {case.z_wall.size} span planes, '
        f'$C_{{p,min}}$ = {case.cp_wall.min():.2f}, $C_{{p,max}}$ = {case.cp_wall.max():.2f}'
    )
    ax.set_xlim(-180.0, 180.0)
    ax.set_xticks(np.arange(-180, 181, 45))

    print(f'wall nodes per plane = {case.n_theta}, planes = {case.z_wall.size}')
    print(f'Cp min = {case.cp_wall.min():.4f}, Cp max = {case.cp_wall.max():.4f}')
    print(f'spanwise Cp spread = {np.ptp(case.cp_wall, axis=0).max():.4f}')

    return save_show(fig, case.fig_dir, 'cp_surface.png')


def plot_cp_surface_time(case, cp_curves):
    if not cp_curves:
        print('no snapshots available, time-resolved Cp is skipped')
        return None

    order = np.argsort(np.degrees(cp_curves[0][2]))
    theta_axis = np.degrees(np.asarray(cp_curves[0][2]))[order]
    cp_time = np.array([np.asarray(curve[1])[order] for curve in cp_curves])
    cp_mean = cp_time.mean(axis=0)
    labels = [curve[0] for curve in cp_curves]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout='constrained')
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cp_curves)))
    for k in range(len(cp_curves)):
        ax.plot(theta_axis, cp_time[k], color=colors[k], linewidth=0.9, label=labels[k])
    ax.fill_between(
        theta_axis,
        cp_time.min(axis=0),
        cp_time.max(axis=0),
        color=C_GRAY,
        alpha=0.25,
        linewidth=0,
        label='min-max over time',
    )
    ax.plot(theta_axis, cp_mean, color=C_RED, linewidth=1.9, label='time mean')
    ax.axhline(0.0, color=C_GUIDE, linewidth=0.7, linestyle='--')
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'$C_p$')
    ax.set_title(
        f'surface $C_p$ over time, span-averaged over [(z/D) {case.mid_lo:.2f}, {case.mid_hi:.2f}], '
        f'{len(cp_curves)} snapshots'
    )
    ax.set_xlim(-180.0, 180.0)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.grid(True, which='major', color=C_GRID, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, fontsize=8.0)

    at_zero = cp_time[:, np.argmin(np.abs(theta_axis))]
    print(f'Cp time-mean: min = {cp_mean.min():.4f} at theta = {theta_axis[np.argmin(cp_mean)]:.1f} deg')
    print(f'Cp time-mean: max = {cp_mean.max():.4f} at theta = {theta_axis[np.argmax(cp_mean)]:.1f} deg')
    print(f'Cp range at theta = 0 deg over time: {at_zero.min():.4f} ... {at_zero.max():.4f}')

    return save_show(fig, case.fig_dir, 'cp_surface_time.png')
