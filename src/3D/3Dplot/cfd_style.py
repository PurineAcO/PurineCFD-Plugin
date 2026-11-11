"""共享绘图风格、配色与保存工具。"""

from pathlib import Path

import matplotlib.pyplot as plt

CMAP_FLOW = 'RdBu_r'
C_RED = '#cc0000'
C_BLUE = '#0000dd'
C_GRAY = '0.72'
C_REF = '0.45'
C_GUIDE = '0.55'
C_TEXT = '0.35'
C_GRID = '0.88'
C_GRID_MINOR = '0.94'

RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.9,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'xtick.minor.size': 2.2,
    'ytick.minor.size': 2.2,
    'axes.grid': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def apply_style():
    plt.rcParams.update(RC)


def repo_root(start=None):
    here = Path.cwd() if start is None else Path(start)
    return next(q for q in (here, *here.parents) if (q / 'pyproject.toml').exists())


def save_show(fig, fig_dir, name):
    path = Path(fig_dir) / name
    fig.savefig(path)
    plt.show()
    print(path)
    return path
