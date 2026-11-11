"""升阻力系数：history 总力与壁面压力积分（纯压差）对比。"""

import matplotlib.pyplot as plt

from cfd_style import C_BLUE, C_GUIDE, C_GRID, C_RED, save_show


def plot_forces(case, force):
    urans = case.urans
    if case.time_axis is not None:
        curve_x = urans['time'].to_numpy(float)
        x_label = 't [s]'
    else:
        curve_x = urans['step'].to_numpy(float)
        x_label = 'iteration'

    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout='constrained')
    ax.plot(curve_x, urans['cl'].to_numpy(float), color=C_BLUE, linewidth=1.0, label=r'$C_l$, total (history)')
    ax.plot(curve_x, urans['cd'].to_numpy(float), color=C_RED, linewidth=1.0, label=r'$C_d$, total (history)')
    ax.plot(
        force['time'],
        force['cl_full'],
        linestyle='none',
        marker='o',
        markersize=5.0,
        color=C_BLUE,
        markeredgecolor='w',
        markeredgewidth=0.7,
        label=r'$C_l$, pressure only (surface)',
    )
    ax.plot(
        force['time'],
        force['cd_full'],
        linestyle='none',
        marker='s',
        markersize=5.0,
        color=C_RED,
        markeredgecolor='w',
        markeredgewidth=0.7,
        label=r'$C_d$, pressure only (surface)',
    )

    ax.axhline(0.0, color=C_GUIDE, linewidth=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel('force coefficient')
    ax.set_title(
        rf'D = {case.diam:.2f} m, Total vs Surface Pressure '
        rf'(Mean $C_d$ = {force["cd_hist"].mean():.3f}, Surface Mean $C_d$ = {force["cd_full"].mean():.3f})'
    )
    ax.set_xlim(float(curve_x.min()), float(curve_x.max()))
    ax.set_ylim(-1.5, 2.0)
    ax.grid(True, which='major', color=C_GRID, linewidth=0.6)
    ax.legend(loc='lower left', ncol=2)

    print(f'D = {case.diam:.4f} m, snapshots used = {len(case.snapshots)}')
    print(f'Mean Cd (history total)  = {force["cd_hist"].mean():.4f}')
    print(f'Mean Cd (pressure only)  = {force["cd_full"].mean():.4f}')
    print(f'Mean friction contribution = {force["friction_cd"].mean():.4f}')
    print(force.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    return save_show(fig, case.fig_dir, 'force_history.png')
