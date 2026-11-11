"""残差收敛曲线：res0 为内迭代残差，res 为总残差。"""

import matplotlib.pyplot as plt
import numpy as np

from cfd_style import C_BLUE, C_GRID, C_GRID_MINOR, C_RED, C_TEXT, save_show


def plot_residual(case):
    steady, urans = case.steady, case.urans
    n_offset = int(steady['step'].max()) if len(steady) else 0
    iteration = np.concatenate(
        (steady['step'].to_numpy(float), urans['step'].to_numpy(float) + n_offset)
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.0), layout='constrained')
    if case.has_inner:
        ax.semilogy(
            iteration,
            np.concatenate((steady['res0'], urans['res0'])),
            color=C_BLUE,
            linewidth=1.0,
            label=r'$res_0$',
        )
    ax.semilogy(
        iteration,
        np.concatenate((steady['res'], urans['res'])),
        color=C_RED,
        linewidth=1.4,
        label=r'$res$',
    )
    if len(steady):
        ax.axvline(n_offset, color=C_TEXT, linewidth=0.8, linestyle='--')
        ax.text(n_offset * 0.5, 0.55, 'steady', transform=ax.get_xaxis_transform(), ha='center', color=C_TEXT)
        ax.text(n_offset * 1.5, 0.55, 'URANS', transform=ax.get_xaxis_transform(), ha='center', color=C_TEXT)

    ax.set_xlabel('iteration')
    ax.set_ylabel('residual')
    ax.set_title(f'{case.name}, $res$ with $iteration$')
    ax.set_xlim(0.0, float(iteration.max()) * 1.02)
    ax.grid(True, which='major', color=C_GRID, linewidth=0.6)
    ax.grid(True, which='minor', color=C_GRID_MINOR, linewidth=0.4)
    ax.legend(loc='upper right')

    print(f'rows = {case.hist.shape[0]}, inner residual = {case.has_inner}')
    print(f'steady rows = {len(steady)}, transient rows = {len(urans)}')
    print(f'res = {urans["res"].iloc[0]:.4e} -> {urans["res"].iloc[-1]:.4e}')
    if case.has_inner:
        print(f'res0 = {urans["res0"].iloc[0]:.4e} -> {urans["res0"].iloc[-1]:.4e}')

    return save_show(fig, case.fig_dir, 'residual.png')
