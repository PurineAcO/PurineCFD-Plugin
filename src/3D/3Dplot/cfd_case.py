"""算例读取：历史量、壁面压力与快照后处理。"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cfd_style import repo_root


@dataclass
class CaseData:
    name: str
    case_dir: Path
    fig_dir: Path
    hist: pd.DataFrame
    steady: pd.DataFrame
    urans: pd.DataFrame
    has_phase: bool
    has_inner: bool
    time_axis: object = None
    wall: pd.DataFrame = None
    z_wall: np.ndarray = None
    n_theta: int = 0
    dtheta: float = 0.0
    radius: float = 0.0
    diam: float = 0.0
    cp_wall: np.ndarray = None
    th_wall: np.ndarray = None
    theta_wall: np.ndarray = None
    mid_lo: float = 0.0
    mid_hi: float = 0.0
    mid_mask: np.ndarray = None
    snapshots: list = field(default_factory=list)


@dataclass
class Snapshots:
    force: pd.DataFrame
    cp_curves: list = field(default_factory=list)
    ref: object = None


def _split_history(hist):
    has_phase = 'phase' in hist.columns
    has_inner = 'res0' in hist.columns and 'inner' in hist.columns
    if has_phase:
        steady = hist[hist['phase'] == 'S']
        urans = hist[hist['phase'] == 'U']
    else:
        steady = hist.iloc[:0]
        urans = hist
    return steady, urans, has_phase, has_inner


def _mid_third(z_wall):
    dz = float(np.median(np.diff(z_wall))) if z_wall.size > 1 else 1.0
    lo_edge = float(z_wall.min() - 0.5 * dz)
    hi_edge = float(z_wall.max() + 0.5 * dz)
    span = hi_edge - lo_edge
    mid_lo = lo_edge + span / 3.0
    mid_hi = hi_edge - span / 3.0
    mask = (z_wall >= mid_lo - 1e-9) & (z_wall <= mid_hi + 1e-9)
    return mid_lo, mid_hi, mask


def load_case(case_name, data_root=None, fig_root=None):
    root = repo_root()
    data_root = root / 'data' if data_root is None else Path(data_root)
    case_dir = data_root / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f'case directory not found: {case_dir}')

    fig_dir = root / 'results' / case_name if fig_root is None else Path(fig_root)
    fig_dir.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(case_dir / 'history.csv')
    steady, urans, has_phase, has_inner = _split_history(hist)

    time_axis = None
    if 'time' in hist.columns and urans['time'].nunique() > 1:
        time_axis = urans.groupby('step')['time'].first()

    wall = pd.read_csv(case_dir / 'wall_cp.csv')
    z_wall = np.unique(wall['z'].to_numpy())
    n_theta = int(wall.shape[0] // z_wall.size)
    dtheta = 2.0 * np.pi / n_theta
    radius = float(np.median(np.hypot(wall['x'], wall['y'])))
    cp_wall = wall['cp'].to_numpy().reshape(z_wall.size, n_theta)
    th_wall = np.arctan2(wall['y'].to_numpy(), wall['x'].to_numpy()).reshape(z_wall.size, n_theta)
    mid_lo, mid_hi, mid_mask = _mid_third(z_wall)

    snapshots = sorted(case_dir.glob('field_[0-9][0-9][0-9][0-9][0-9].csv'))
    if not snapshots and (case_dir / 'field.csv').exists():
        snapshots = [case_dir / 'field.csv']

    return CaseData(
        name=case_name,
        case_dir=case_dir,
        fig_dir=fig_dir,
        hist=hist,
        steady=steady,
        urans=urans,
        has_phase=has_phase,
        has_inner=has_inner,
        time_axis=time_axis,
        wall=wall,
        z_wall=z_wall,
        n_theta=n_theta,
        dtheta=dtheta,
        radius=radius,
        diam=2.0 * radius,
        cp_wall=cp_wall,
        th_wall=th_wall,
        theta_wall=np.degrees(th_wall[0]),
        mid_lo=mid_lo,
        mid_hi=mid_hi,
        mid_mask=mid_mask,
        snapshots=snapshots,
    )


def _snapshot_step(path, fallback):
    digits = [int(q) for q in path.stem.split('_')[1:] if q.isdigit()]
    return digits[0] if digits else fallback


def load_snapshots(case):
    rows = []
    curves = []
    ref = None
    fallback_step = int(case.hist['step'].max())

    for path in case.snapshots:
        snap = pd.read_csv(path)
        xs = snap['x'].to_numpy()
        ys = snap['y'].to_numpy()
        zs = snap['z'].to_numpy()
        ps = snap['p'].to_numpy()
        rs = np.hypot(xs, ys)
        ts = np.arctan2(ys, xs)

        far = rs > 0.7 * rs.max()
        p_ref = float(np.median(ps[far]))
        rho_ref = float(np.median(snap['rho'].to_numpy()[far]))
        u_ref = float(np.median(snap['u'].to_numpy()[far]))
        q_ref = 0.5 * rho_ref * u_ref ** 2
        ref = {'p_inf': p_ref, 'rho_inf': rho_ref, 'u_inf': u_ref, 'q_inf': q_ref}

        inner = np.isclose(rs, rs.min())
        z_snap = np.unique(zs[inner])
        n_t = int(inner.sum() // z_snap.size)
        cp = ((ps[inner] - p_ref) / q_ref).reshape(z_snap.size, n_t)
        th = ts[inner].reshape(z_snap.size, n_t)

        dtheta = 2.0 * np.pi / n_t
        cd_planes = -(cp * np.cos(th)).sum(axis=1) * case.radius * dtheta / case.diam
        cl_planes = -(cp * np.sin(th)).sum(axis=1) * case.radius * dtheta / case.diam

        step = _snapshot_step(path, fallback_step)
        if case.time_axis is not None and step in case.time_axis.index:
            moment = float(case.time_axis.loc[step])
        else:
            moment = float(step)

        hist_rows = case.urans.loc[case.urans['step'] == step, 'cd']
        cd_hist = float(hist_rows.iloc[-1]) if len(hist_rows) else float('nan')

        rows.append(
            {
                'step': step,
                'time': moment,
                'cd_full': float(cd_planes.mean()),
                'cl_full': float(cl_planes.mean()),
                'cd_hist': cd_hist,
                'friction_cd': cd_hist - float(cd_planes.mean()),
            }
        )

        _, _, mid_mask = _mid_third(z_snap)
        label = f't = {moment:.3f} s' if case.time_axis is not None else f'step {step}'
        curves.append((label, cp[mid_mask].mean(axis=0), th[0]))

    force = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    return Snapshots(force=force, cp_curves=curves, ref=ref)
