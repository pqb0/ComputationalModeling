
"""
Laplace solver for a square domain with a circular Dirichlet conductor.
Implements Jacobi, Gauss-Seidel and SOR.
All tasks of the assignment are performed.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List

#  Solver class

class Method:
    def __init__(self,
                 N: int,
                 r_cond: float = 0.2,
                 V_cond: float = 1.0,
                 V_boundary: float = 0.0,
                 method: str = 'jacobi',
                 omega: float = None):
        self.N = N
        self.h = 1.0 / (N - 1)

        x = np.linspace(0.0, 1.0, N)
        y = np.linspace(0.0, 1.0, N)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        self.phi = np.full((N, N), 0.0, dtype=float)

        # Dirichlet on outer sides
        self.phi[0, :] = V_boundary
        self.phi[-1, :] = V_boundary
        self.phi[:, 0] = V_boundary
        self.phi[:, -1] = V_boundary

        # circular conductor mask
        dist2 = (self.X - 0.5)**2 + (self.Y - 0.5)**2
        self.mask_cond = dist2 <= r_cond**2
        self.phi[self.mask_cond] = V_cond

        # points that will be updated each sweep
        interior = np.ones_like(self.phi, dtype=bool)
        interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
        self.update_mask = interior & (~self.mask_cond)

        self.method = method.lower()
        if self.method not in {'jacobi', 'gauss-seidel', 'sor'}:
            raise ValueError('method must be jacobi, gauss-seidel or sor')

        if self.method == 'sor':
            self.omega = omega if omega is not None else 2.0 / (1.0 + (np.pi / self.N))
        else:
            self.omega = None

    # ----------------------------------------------------------
    #  single-sweep helpers
    # ----------------------------------------------------------
    def _jacobi_sweep(self, phi_old):
        phi_new = phi_old.copy()
        i, j = np.where(self.update_mask)
        phi_new[i, j] = 0.25 * (
            phi_old[i + 1, j] + phi_old[i - 1, j] +
            phi_old[i, j + 1] + phi_old[i, j - 1]
        )
        return phi_new

    def _gauss_seidel_sweep(self, phi):
        N = self.N
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if self.update_mask[i, j]:
                    phi[i, j] = 0.25 * (
                        phi[i + 1, j] + phi[i - 1, j] +
                        phi[i, j + 1] + phi[i, j - 1]
                    )
        return phi

    def _sor_sweep(self, phi):
        omega = self.omega
        N = self.N
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if self.update_mask[i, j]:
                    phi_GS = 0.25 * (
                        phi[i + 1, j] + phi[i - 1, j] +
                        phi[i, j + 1] + phi[i, j - 1]
                    )
                    phi[i, j] = (1.0 - omega) * phi[i, j] + omega * phi_GS
        return phi

    #  public run method



    def run(self, tol: float = 1e-6, max_iter: int = 200000):
        phi = self.phi.copy()
        history = []


        start = time.perf_counter()
        for k in range(max_iter):
            if self.method == 'jacobi':
                phi_new = self._jacobi_sweep(phi)
                diff = np.max(np.abs(phi_new - phi))
                phi = phi_new
            elif self.method == 'gauss-seidel':
                phi_old = phi.copy()
                phi = self._gauss_seidel_sweep(phi)
                diff = np.max(np.abs(phi - phi_old))

            else:   # SOR
                phi_old = phi.copy()
                phi = self._sor_sweep(phi)
                diff = np.max(np.abs(phi - phi_old))


            history.append(diff)
            if diff < tol:
                break

        elapsed = time.perf_counter() - start
        omega_str = '-' if self.omega is None else f'{self.omega:5.2f}'
        print(f'{self.method.upper():12s}  N={self.N:3d}  ω={omega_str:5s} '
              f'→ {k+1:5d} iters, {elapsed:6.3f}s, final Δ={diff:.2e}')
        return phi, history


def electric_field(phi, h):
    Ex = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * h)
    Ey = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * h)
    Ex[[0, -1], :] = Ey[:, [0, -1]] = 0.0
    return Ex, Ey


def plot_potential(phi, X, Y, title, fname):
    plt.figure(figsize=(6,5))
    cp = plt.contourf(X, Y, phi, levels=30, cmap='viridis')
    plt.colorbar(cp, label='Potential')
    plt.contour(X, Y, phi, levels=[0.5], colors='k', linewidths=0.5)
    plt.title(title); plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
    plt.savefig(fname, dpi=150); plt.close()


def plot_field(Ex, Ey, X, Y, title, fname, stride=5):
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, np.sqrt(Ex**2+Ey**2), cmap='plasma', levels=30)
    plt.colorbar(label='|E|')
    plt.quiver(X[::stride, ::stride], Y[::stride, ::stride],
               Ex[::stride, ::stride], Ey[::stride, ::stride],
               color='w', scale=30)
    plt.title(title); plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
    plt.savefig(fname, dpi=150); plt.close()


def plot_convergence(hist_dict, title, fname):
    plt.figure(figsize=(6,4))
    for lab, hist in hist_dict.items():
        plt.semilogy(range(1, len(hist)+1), hist, label=lab)
    plt.xlabel('Iteration'); plt.ylabel('max |Δφ|')
    plt.title(title); plt.grid(True, which='both'); plt.legend()
    plt.savefig(fname, dpi=150); plt.close()


def main():

    N = 101
    tol = 1e-6
    print('\n' + '='*60)
    print(f'   GRID SIZE N = {N}')
    print('='*60)

    potentials, histories = {}, {}

     # Jacobi
    jac = Method(N=N, method='jacobi')
    phi_jac, hist_jac = jac.run(tol=tol)
    potentials['Jacobi'] = phi_jac
    histories['Jacobi'] = hist_jac

    # Gauss-Seidel
    gs = Method(N=N, method='gauss-seidel')
    phi_gs, hist_gs = gs.run(tol=tol)
    potentials['Gauss-Seidel'] = phi_gs
    histories['Gauss-Seidel'] = hist_gs

    # SOR - test several ω
    omega_opt = 2.0 / (1.0 + np.sin(np.pi * gs.h))
    test_omegas = [1.0, 1.2, 1.5, omega_opt, 1.99]
    for w in test_omegas:
            label = f'SOR ω={w:.3f}'
            sor = Method(N=N, method='SOR', omega=w)
            phi_sor, hist_sor = sor.run(tol=tol)
            potentials[label] = phi_sor
            histories[label] = hist_sor

    # ---- plotting -------------------------------------------------
    # best SOR solution (optimal ω)
    phi_best = potentials[f'SOR ω={omega_opt:.3f}']
    title = f'Potential (N={N}) - SOR ω={omega_opt:.3f}'
    plot_potential(phi_best, jac.X, jac.Y, title, f'potential_N{N}.png')

    Ex, Ey = electric_field(phi_best, jac.h)
    title = f'Electric field (N={N}) - SOR ω={omega_opt:.3f}'
    plot_field(Ex, Ey, jac.X, jac.Y, title, f'field_N{N}.png',
            stride=max(1, N//30))

    # convergence histories (Jacobi, Gauss-Seidel, optimal SOR)
    short_hist = {
        'Jacobi': histories['Jacobi'],
        'Gauss-Seidel': histories['Gauss-Seidel'],
        f'SOR ω={omega_opt:.3f}': histories[f'SOR ω={omega_opt:.3f}']
    }
    plot_convergence(short_hist,
                 f'Convergence (max-change) - N={N}',
                 f'convergence_N{N}.png')

  
    #  boundary=1 experiment

    print('\n' + '='*60)
    print('   BOUNDARY TEST - outer sides set to V = 1')
    print('='*60)

    N = 101
    bnd = Method(N=N, V_boundary=1.0, method='SOR')
    phi_b, _ = bnd.run(tol=tol)
    plot_potential(phi_b, bnd.X, bnd.Y,
                   f'Potential with V=1 on all sides (N={N})',
                   'potential_boundary1.png')

    print('\nAll figures have been saved to the current directory.')


if __name__ == '__main__':
    main()
