import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class flowsolve:
    def __init__(self,M0,T0,p_out,L,N,CFL=1.0,t_end=4.0e-3,snap_t=None):

        self.gamma = 1.4
        self.Rgas = 287.0
        self.pulse = 0.02
        self.pulse_width = 0.03

        self.M0 = float(M0)
        self.T0 = float(T0)
        self.p_out = float(p_out)
        self.L = float(L)
        self.N = int(N)
        self.CFL = float(CFL)
        self.t_end = float(t_end)

        if snap_t is None:
            snap_t = (0.0, 0.3e-3, 0.6e-3, 0.9e-3, 1.2e-3, 2.0e-3, 3.0e-3, 4.0e-3)
        self.snap_t = [float(s) for s in sorted(snap_t)]

        self.dx = self.L / self.N
        self.x = (np.arange(self.N) + 0.5) * self.dx
        self.c0 = np.sqrt(self.gamma * self.Rgas * self.T0)
        self.u0 = self.M0 * self.c0
        self.rho0 = self.p_out / (self.Rgas * self.T0)
        self.dt = self.CFL * self.dx / (self.u0 + self.c0)

        self.U = None
        self.t = 0.0
        self.snaps = []
        self.steps = 0

    
    def prim(self, U):
        U = np.asarray(U)
        rho, mom, E = (U[0], U[1], U[2]) if U.ndim == 1 else (U[:, 0], U[:, 1], U[:, 2])
        u = mom / rho
        p = (self.gamma - 1.0) * (E - 0.5 * rho * u * u)
        return rho, u, p

    def cons(self, rho, u, p):
        return np.array([rho, rho * u, p / (self.gamma - 1.0) + 0.5 * rho * u * u])

    def flux(self, U):
        rho, u, p = self.prim(U)
        E = U[2] if np.ndim(U) == 1 else U[:, 2]
        return np.stack([rho * u, rho * u * u + p, u * (E + p)], axis=-1)

    def euler_jac(self, U):
        gamma = self.gamma
        rho, u, p = self.prim(U)
        H = (U[2] + p) / rho
        return np.array([
            [0.0, 1.0, 0.0],
            [-0.5 * (3.0 - gamma) * u * u, (3.0 - gamma) * u, gamma - 1.0],
            [u * (0.5 * (gamma - 1.0) * u * u - H),
             H - (gamma - 1.0) * u * u,
             gamma * u],
        ])

    def central_flux(self, UL, UR):
        return 0.5 * (self.flux(UL) + self.flux(UR))

    def make_ghosts(self, U):
        gamma, Rgas, T0, p_out = self.gamma, self.Rgas, self.T0, self.p_out

        # Velocity Inlet
        rhoL = U[0, 0]
        pL = rhoL * Rgas * T0
        Ug = self.cons(rhoL, self.u0, pL)

        # Pressure Outlet
        rhoR = U[-1, 0]
        uR = U[-1, 1] / rhoR
        Ugo = self.cons(rhoR, uR, p_out)

        # GL = dUg/dU0
        vL = np.array([1.0, self.u0, Rgas * T0 / (gamma - 1.0) + 0.5 * self.u0 * self.u0])
        GL = np.zeros((3, 3))
        GL[:, 0] = vL

        # GR = dUgo/dU_{N-1}
        GR = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.5 * uR * uR, uR, 0.0],
        ])
        return Ug, Ugo, GL, GR

    def initial_field(self):
        gamma = self.gamma
        U = np.empty((self.N, 3))
        U[:] = self.cons(self.rho0, self.u0, self.p_out)

        pert = self.pulse * np.exp(-((self.x - 0.5 * self.L) / (self.pulse_width * self.L)) ** 2)
        p = self.p_out * (1.0 + pert)
        rho = self.rho0 * (p / self.p_out) ** (1.0 / gamma)  # 等熵扰动
        for i in range(self.N):
            U[i] = self.cons(rho[i], self.u0, p[i])
        return U

    def residual(self, U):
        ghosts = self.make_ghosts(U)
        Ug, Ugo = ghosts[0], ghosts[1]
        F = np.empty((self.N + 1, 3))
        F[0] = self.central_flux(Ug, U[0])
        for j in range(1, self.N):
            F[j] = self.central_flux(U[j - 1], U[j])
        F[self.N] = self.central_flux(U[self.N - 1], Ugo)
        return (F[:-1] - F[1:]) / self.dx, ghosts

    def assemble(self, U, ghosts, dt):
        Ug, Ugo, GL, GR = ghosts
        N = self.N
        rows, cols, vals = [], [], []

        def add_block(i, j, B):
            for a in range(3):
                for b in range(3):
                    rows.append(3 * i + a)
                    cols.append(3 * j + b)
                    vals.append(B[a, b])

        I3 = np.eye(3)
        for i in range(N):
            Aii = I3 / dt
            if i == 0:
                Aii = Aii - 0.5 * self.euler_jac(Ug) @ GL / self.dx
            elif i == N - 1:
                Aii = Aii + 0.5 * self.euler_jac(Ugo) @ GR / self.dx
            add_block(i, i, Aii)

            if i > 0:
                add_block(i, i - 1, -0.5 * self.euler_jac(U[i - 1]) / self.dx)
            if i < N - 1:
                add_block(i, i + 1, 0.5 * self.euler_jac(U[i + 1]) / self.dx)

        return sp.csr_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N))

    def step(self, U, dt):
        Res, ghosts = self.residual(U)
        Mat = self.assemble(U, ghosts, dt)
        dU = spla.spsolve(Mat, Res.reshape(-1)).reshape(self.N, 3)
        return U + dU

    def solve(self, t_end=None, snap_t=None, verbose=True):
        if t_end is not None:
            self.t_end = float(t_end)
        if snap_t is not None:
            self.snap_t = [float(s) for s in sorted(snap_t)]

        U = self.initial_field()
        t = 0.0
        dt = self.dt
        snaps = [(0.0, self.x.copy(), self.prim(U)[2])]
        next_snap = 1
        steps = 0

        while t < self.t_end - 1e-14:
            dt = min(dt, self.t_end - t)
            U = self.step(U, dt)
            t += dt
            steps += 1

            while next_snap < len(self.snap_t) and t >= self.snap_t[next_snap] - 1e-14:
                snaps.append((t, self.x.copy(), self.prim(U)[2]))
                next_snap += 1

        self.U = U
        self.t = t
        self.snaps = snaps
        self.steps = steps

        if verbose:
            print(f'N = {self.N}, dx = {self.dx:.4e} m, dt = {self.dt:.4e} s, steps = {self.steps}')
            print(f't = {self.t:.4e} s, snapshots = {len(self.snaps)}')
        return self

    def results(self):
        if self.U is None:
            raise RuntimeError('call solve() before results()')

        rho, u, p = self.prim(self.U)
        T = p / (rho * self.Rgas)
        return {
            'x': self.x,
            't': np.array([s[0] for s in self.snaps]),
            'snap_x': [s[1] for s in self.snaps],
            'snap_p': [s[2] for s in self.snaps],
            'snap_dp': [(s[2] - self.p_out) / self.p_out * 100.0 for s in self.snaps],
            'rho': rho,
            'u': u,
            'p': p,
            'T': T,
            'M': u / np.sqrt(self.gamma * self.Rgas * T),
            'dt': self.dt,
            'steps': self.steps,
        }

    def plot(self, save_path=None, show=True):
        data = self.results()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for xs, dp, ts in zip(data['snap_x'], data['snap_dp'], data['t']):
            ax.plot(xs, dp, label=f't={ts * 1e3:.1f} ms')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('(p - p_out)/p_out  [%]')
        ax.set_title(
            '1D compressible pipe: implicit Euler + central flux '
            f'(M0={self.M0:g}, N={self.N}, CFL={self.CFL:g})'
        )
        ax.legend(ncol=2)
        ax.grid(True)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f'saved: {save_path}')
        if show:
            plt.show()
