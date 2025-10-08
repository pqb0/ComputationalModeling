#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical study of a charged particle (or two particles) in static
electric and magnetic fields.  The code contains:

*   Method                – single‑particle pusher (Euler / Euler‑Richardson)
*   TwoParticleMethod     – same but adds the Coulomb interaction
*   Analytic solutions   – E‑field, B‑field, crossed fields
*   Test suites          – tasks (b)–(e) from the assignment
*   Plotting utilities   – 1‑D, 3‑D, convergence & energy diagnostics

The structure mirrors the original script you provided; only small
renamings and extra helper methods were introduced.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from typing import Tuple, List, Callable

# ----------------------------------------------------------------------
# 1️⃣  SINGLE‑PARTICLE CLASS (Euler / Euler‑Richardson)
# ----------------------------------------------------------------------
class Method:
    """
    Integrates the Lorentz‑force equation for a single charged particle.

    Parameters
    ----------
    t0 : float
        Initial time.
    r0 : array‑like, shape (3,)
        Initial position.
    v0 : array‑like, shape (3,)
        Initial velocity.
    dt : float
        Time step.
    t_end : float
        Final integration time.
    q, m : float
        Charge and mass.
    E, B : array‑like, shape (3,)
        Constant electric and magnetic fields.
    method : str, optional
        'euler' or 'euler‑richardson'.
    """
    def __init__(self,
                 t0: float,
                 r0: np.ndarray,
                 v0: np.ndarray,
                 dt: float,
                 t_end: float,
                 q: float,
                 m: float,
                 E: np.ndarray,
                 B: np.ndarray,
                 method: str = 'euler'):

        self.t0   = t0
        self.r0   = np.asarray(r0, dtype=float)
        self.v0   = np.asarray(v0, dtype=float)
        self.dt   = dt
        self.t_end = t_end
        self.q    = q
        self.m    = m
        self.E    = np.asarray(E, dtype=float)
        self.B    = np.asarray(B, dtype=float)
        self.method = method.lower()
        self.q_over_m = q / m

    # --------- Lorentz acceleration (constant fields) ----------
    def _acceleration(self, v: np.ndarray) -> np.ndarray:
        """a = (q/m)(E + v × B)   (E and B are constant)."""
        return self.q_over_m * (self.E + np.cross(v, self.B))

    # --------- Main integration routine ----------
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate and return (t, r(t), v(t))."""
        t_vals = np.arange(self.t0,
                           self.t_end + self.dt,
                           self.dt)
        n = len(t_vals)

        r = np.zeros((n, 3))
        v = np.zeros((n, 3))

        r[0] = self.r0
        v[0] = self.v0

        for i in range(n - 1):
            if self.method == 'euler':
                r[i + 1] = r[i] + self.dt * v[i]
                v[i + 1] = v[i] + self.dt * self._acceleration(v[i])

            elif self.method == 'euler-richardson':
                # half–step (mid‑point) quantities
                v_mid = v[i] + 0.5 * self.dt * self._acceleration(v[i])
                r_mid = r[i] + 0.5 * self.dt * v[i]

                r[i + 1] = r[i] + self.dt * v_mid
                v[i + 1] = v[i] + self.dt * self._acceleration(v_mid)

            else:
                raise ValueError("method must be 'euler' or 'euler-richardson'")

        return t_vals, r, v

# ----------------------------------------------------------------------
# 2️⃣  TWO‑PARTICLE CLASS (adds Coulomb interaction)
# ----------------------------------------------------------------------
class TwoParticleMethod:
    """
    Simultaneous integration of two charged particles.
    The particles feel the same external static E, B fields and the
    pair‑wise Coulomb force.
    """
    epsilon0 = 8.854187817e-12   # SI, can be set to 1 for normalized units

    def __init__(self,
                 t0: float,
                 r1_0: np.ndarray, v1_0: np.ndarray,
                 r2_0: np.ndarray, v2_0: np.ndarray,
                 dt: float,
                 t_end: float,
                 q1: float, m1: float,
                 q2: float, m2: float,
                 E: np.ndarray,
                 B: np.ndarray,
                 method: str = 'euler-richardson'):

        self.t0, self.dt, self.t_end = t0, dt, t_end
        self.r1_0, self.v1_0 = np.asarray(r1_0, float), np.asarray(v1_0, float)
        self.r2_0, self.v2_0 = np.asarray(r2_0, float), np.asarray(v2_0, float)
        self.q1, self.m1 = q1, m1
        self.q2, self.m2 = q2, m2
        self.E = np.asarray(E, float)
        self.B = np.asarray(B, float)
        self.method = method.lower()

    # ---------- Lorentz part (same as in Method) ----------
    def _acc_lorentz(self, q, m, v):
        return (q / m) * (self.E + np.cross(v, self.B))

    # ---------- Coulomb acceleration exerted by particle j on i ----------
    def _acc_coulomb(self, q_i, q_j, r_i, r_j):
        r_vec = r_i - r_j
        r_norm = np.linalg.norm(r_vec)
        if r_norm == 0.0:
            return np.zeros(3)                # avoid division by zero
        return (q_i * q_j) / (self.epsilon0 * r_norm ** 3) * r_vec

    # ---------- Full step for one particle ----------
    def _step(self, r, v, q, m, r_other, v_other):
        """Return (r_next, v_next) for a single particle."""
        a_lor = self._acc_lorentz(q, m, v)
        a_coul = self._acc_coulomb(q, q_other, r, r_other)  # q_other defined later

    # ---------- Integrator ----------
    def run(self) -> Tuple[np.ndarray,
                           np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray]:
        """Integrate both particles, return (t, r1, v1, r2, v2)."""
        t_vals = np.arange(self.t0,
                           self.t_end + self.dt,
                           self.dt)
        n = len(t_vals)

        r1 = np.zeros((n, 3))
        v1 = np.zeros((n, 3))
        r2 = np.zeros((n, 3))
        v2 = np.zeros((n, 3))

        r1[0], v1[0] = self.r1_0, self.v1_0
        r2[0], v2[0] = self.r2_0, self.v2_0

        for i in range(n - 1):
            # ---------- Euler ----------
            if self.method == 'euler':
                a1 = self._acc_lorentz(self.q1, self.m1, v1[i]) + \
                     self._acc_coulomb(self.q1, self.q2, r1[i], r2[i])
                a2 = self._acc_lorentz(self.q2, self.m2, v2[i]) + \
                     self._acc_coulomb(self.q2, self.q1, r2[i], r1[i])

                r1[i + 1] = r1[i] + self.dt * v1[i]
                v1[i + 1] = v1[i] + self.dt * a1

                r2[i + 1] = r2[i] + self.dt * v2[i]
                v2[i + 1] = v2[i] + self.dt * a2

            # ---------- Euler‑Richardson ----------
            elif self.method == 'euler-richardson':
                # ---- half‑step for particle 1 ----
                a1_full = self._acc_lorentz(self.q1, self.m1, v1[i]) + \
                          self._acc_coulomb(self.q1, self.q2, r1[i], r2[i])
                v1_mid = v1[i] + 0.5 * self.dt * a1_full
                r1_mid = r1[i] + 0.5 * self.dt * v1[i]

                # ---- half‑step for particle 2 (needs the half‑step r1) ----
                a2_full = self._acc_lorentz(self.q2, self.m2, v2[i]) + \
                          self._acc_coulomb(self.q2, self.q1, r2[i], r1[i])
                v2_mid = v2[i] + 0.5 * self.dt * a2_full
                r2_mid = r2[i] + 0.5 * self.dt * v2[i]

                # ---- accelerations evaluated at the mid‑point ----
                a1_mid = self._acc_lorentz(self.q1, self.m1, v1_mid) + \
                         self._acc_coulomb(self.q1, self.q2, r1_mid, r2_mid)
                a2_mid = self._acc_lorentz(self.q2, self.m2, v2_mid) + \
                         self._acc_coulomb(self.q2, self.q1, r2_mid, r1_mid)

                # ---- full update ----
                r1[i + 1] = r1[i] + self.dt * v1_mid
                v1[i + 1] = v1[i] + self.dt * a1_mid

                r2[i + 1] = r2[i] + self.dt * v2_mid
                v2[i + 1] = v2[i] + self.dt * a2_mid

            else:
                raise ValueError("method must be 'euler' or 'euler-richardson'")

        return t_vals, r1, v1, r2, v2

# ----------------------------------------------------------------------
# 3️⃣  ANALYTIC SOLUTIONS (used for validation)
# ----------------------------------------------------------------------
def analytic_electric(t: np.ndarray,
                     r0: np.ndarray,
                     v0: np.ndarray,
                     q: float,
                     m: float,
                     E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic position & velocity for a constant electric field."""
    a = (q / m) * E
    r = r0 + v0[np.newaxis, :] * t[:, np.newaxis] + 0.5 * a[np.newaxis, :] * t[:, np.newaxis] ** 2
    v = v0 + a * t[:, np.newaxis]
    return r, v


def analytic_magnetic(t: np.ndarray,
                      r0: np.ndarray,
                      v0: np.ndarray,
                      q: float,
                      m: float,
                      B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic solution for a uniform magnetic field (no E)."""
    Bnorm = np.linalg.norm(B)
    if Bnorm == 0.0:
        # pure straight line
        r = r0 + v0 * t[:, np.newaxis]
        v = np.tile(v0, (len(t), 1))
        return r, v

    # Decompose velocity into parallel & perpendicular parts
    bhat = B / Bnorm
    v_par = np.dot(v0, bhat) * bhat
    v_perp = v0 - v_par
    v_perp_norm = np.linalg.norm(v_perp)

    omega = q * Bnorm / m                     # cyclotron frequency (sign kept)
    rL = m * v_perp_norm / (abs(q) * Bnorm)   # Larmor radius

    # Choose orthonormal basis (e1, e2) in the plane perpendicular to B
    if v_perp_norm == 0:
        e1 = np.zeros(3)
    else:
        e1 = v_perp / v_perp_norm
    e2 = np.cross(bhat, e1)

    # Phase at t=0
    phi0 = 0.0

    # Position of the guiding centre
    r_gc = r0 + np.cross(v_par, bhat) * (m / (q * Bnorm))

    # Build arrays
    r = np.empty((len(t), 3))
    v = np.empty((len(t), 3))

    for i, ti in enumerate(t):
        # circular motion in the perpendicular plane
        perp = rL * (np.cos(omega * ti + phi0) * e1 +
                     np.sin(omega * ti + phi0) * e2)

        r[i] = r_gc + v_par * ti + perp
        v[i] = v_par + v_perp_norm * (-np.sin(omega * ti + phi0) * e1 +
                                      np.cos(omega * ti + phi0) * e2)

    return r, v


def drift_velocity(E: np.ndarray, B: np.ndarray) -> np.ndarray:
    """E×B drift (valid for any charge sign)."""
    B2 = np.dot(B, B)
    if B2 == 0:
        return np.zeros(3)
    return np.cross(E, B) / B2

# ----------------------------------------------------------------------
# 4️⃣  PLOTTING HELPERS
# ----------------------------------------------------------------------
def plot_3d_trajectory(r_euler, r_rich, title='Particle trajectory'):
    """3‑D overlay of Euler and Euler‑Richardson trajectories."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_euler[:, 0], r_euler[:, 1], r_euler[:, 2],
            label='Euler', color='C0')
    ax.plot(r_rich[:, 0], r_rich[:, 1], r_rich[:, 2],
            label='Euler‑Richardson', color='C1')
    ax.scatter(r_euler[0, 0], r_euler[0, 1], r_euler[0, 2],
               color='k', s=40, label='Start')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_1d(t, y_euler, y_rich, ylabel='x(t)', title='1‑D comparison'):
    plt.figure(figsize=(8, 4))
    plt.plot(t, y_euler, label='Euler', lw=2)
    plt.plot(t, y_rich, label='Euler‑Richardson', lw=2, ls='--')
    plt.xlabel('time')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_energy(t, KE_euler, KE_rich, title='Kinetic energy'):
    plt.figure(figsize=(8, 4))
    plt.plot(t, KE_euler, label='Euler')
    plt.plot(t, KE_rich, label='Euler‑Richardson', ls='--')
    plt.xlabel('time')
    plt.ylabel('K [J]')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_convergence(dt_vals, err_euler, err_rich):
    plt.figure(figsize=(7, 5))
    plt.loglog(dt_vals, err_euler, 'o-', label='Euler')
    plt.loglog(dt_vals, err_rich, 's-', label='Euler‑Richardson')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel('max position error')
    plt.title('Convergence test (electric field)')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


def plot_distance(t, d, title='Inter‑particle distance'):
    plt.figure(figsize=(8, 4))
    plt.plot(t, d, label='|r1–r2|')
    plt.xlabel('time')
    plt.ylabel('distance')
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_total_energy(t, E_tot, title='Total mechanical energy (two particles)'):
    plt.figure(figsize=(8, 4))
    plt.plot(t, E_tot, label='E_tot')
    plt.xlabel('time')
    plt.ylabel('Energy [J]')
    plt.title(title)
    plt.grid(True)
    plt.show()


# ----------------------------------------------------------------------
# 5️⃣  TEST SUITES (tasks b–e)
# ----------------------------------------------------------------------
def run_electric_test():
    """Task (b): pure E‑field, comparison with analytics, convergence."""
    # ------------------------------------------------------------
    # Physical data (normalized to avoid tiny numbers)
    # ------------------------------------------------------------
    q = -1.0          # electron‑like charge
    m = 1.0           # unit mass

    # three sensible field / initial‑velocity combos
    field_set = [
        (np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([1.0, 0.5, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 1.5, -0.2]), np.array([0.0, 0.5, 0.0]))
    ]

    dt_vals = [0.2, 0.1, 0.05, 0.02, 0.01]   # will be used for convergence
    t_end = 8.0
    r0 = np.zeros(3)

    for idx, (E, v0) in enumerate(field_set, 1):
        print(f'\n--- Electric test #{idx}  E={E}  v0={v0} ---')
        # ------ analytic reference -------
        t_ref = np.linspace(0, t_end, 1000)
        r_ref, v_ref = analytic_electric(t_ref, r0, v0, q, m, E)

        # ------ convergence loop -------
        max_err_euler = []
        max_err_rich  = []
        for dt in dt_vals:
            # ----- Euler integration -----
            meth_e = Method(0, r0, v0, dt, t_end, q, m, E,
                            np.zeros(3), method='euler')
            t_e, r_e, _ = meth_e.run()

            # ----- Euler‑Richardson integration -----
            meth_r = Method(0, r0, v0, dt, t_end, q, m, E,
                            np.zeros(3), method='euler-richardson')
            t_r, r_r, _ = meth_r.run()

            # ---- ** NEW: evaluate the analytical solution at the exact
            #      times returned by the integrators (no interpolation) ----
            # The analytic helper already returns both position and velocity,
            # but we only need the position here.
            r_ana_e, _ = analytic_electric(t_e, r0, v0, q, m, E)
            r_ana_r, _ = analytic_electric(t_r, r0, v0, q, m, E)

            # ---- maximum absolute position error for each method ----
            err_e = np.linalg.norm(r_e - r_ana_e, axis=1).max()
            err_r = np.linalg.norm(r_r - r_ana_r, axis=1).max()
            max_err_euler.append(err_e)
            max_err_rich.append(err_r)

        # ------ show convergence plot ------
        plot_convergence(dt_vals, max_err_euler, max_err_rich)

        # ------ pick a “good” dt (the smallest that gives ≤1 % error) ------
        good_dt = dt_vals[-1]      # last (smallest) entry – you can adapt this
        meth_e = Method(0, r0, v0, good_dt, t_end, q, m, E, np.zeros(3), 'euler')
        t_e, r_e, v_e = meth_e.run()
        meth_r = Method(0, r0, v0, good_dt, t_end, q, m, E, np.zeros(3), 'euler-richardson')
        t_r, r_r, v_r = meth_r.run()

        # ------ position 1‑D plots (choose x‑component) ------
        plot_1d(t_e, r_e[:, 0], r_r[:, 0],
                ylabel='x(t)', title=f'Pure E‑field – test #{idx}')

        # ------ kinetic‑energy plot ------
        KE_e = 0.5 * m * np.sum(v_e ** 2, axis=1)
        KE_r = 0.5 * m * np.sum(v_r ** 2, axis=1)
        plot_energy(t_e, KE_e, KE_r,
                    title=f'Kinetic energy (pure E) – test #{idx}')


def run_magnetic_test():
    """Task (c): pure B‑field, circular motion & kinetic‑energy constancy."""
    q = -1.0
    m = 1.0
    B = np.array([0.0, 0.0, 0.5])           # constant field along +z
    E = np.zeros(3)

    # three initial velocities (perpendicular, parallel, mixed)
    vel_set = [
        np.array([1.0, 0.0, 0.0]),          # pure x (perpendicular)
        np.array([0.0, 0.0, 0.5]),          # pure z (parallel)
        np.array([1.0, 0.0, 0.5])           # mixed
    ]

    dt_vals = [0.05, 0.02, 0.01, 0.005]
    t_end = 20.0
    r0 = np.zeros(3)

    for idx, v0 in enumerate(vel_set, 1):
        print(f'\n--- Magnetic test #{idx}  v0={v0} ---')
        # ------------------------------------------------------------
        # Analytic reference (used only for the radius)
        # ------------------------------------------------------------
        w_c = q * np.linalg.norm(B) / m
        rL = m * np.linalg.norm(v0 - np.dot(v0, B) / np.linalg.norm(B) ** 2 * B) / \
             (abs(q) * np.linalg.norm(B))

        # ------------------------------------------------------------
        # Convergence study (radius drift & kinetic‑energy drift)
        # ------------------------------------------------------------
        rad_err_eul = []
        rad_err_rich = []
        ke_err_eul = []
        ke_err_rich = []

        for dt in dt_vals:
            # Euler
            meth_e = Method(0, r0, v0, dt, t_end, q, m, E, B, 'euler')
            t_e, r_e, v_e = meth_e.run()
            # Richardson
            meth_r = Method(0, r0, v0, dt, t_end, q, m, E, B, 'euler-richardson')
            t_r, r_r, v_r = meth_r.run()

            # radius at final time (distance from z‑axis)
            rad_e = np.sqrt(r_e[:, 0] ** 2 + r_e[:, 1] ** 2)
            rad_r = np.sqrt(r_r[:, 0] ** 2 + r_r[:, 1] ** 2)
            rad_err_eul.append(np.abs(rad_e[-1] - rL))
            rad_err_rich.append(np.abs(rad_r[-1] - rL))

            # kinetic‑energy drift (relative to initial KE)
            ke0 = 0.5 * m * np.dot(v0, v0)
            ke_e = 0.5 * m * np.sum(v_e ** 2, axis=1)
            ke_r = 0.5 * m * np.sum(v_r ** 2, axis=1)
            ke_err_eul.append(np.abs(ke_e[-1] - ke0) / ke0)
            ke_err_rich.append(np.abs(ke_r[-1] - ke0) / ke0)

        # ----- convergence plots -----
        plot_convergence(dt_vals, rad_err_eul, rad_err_rich)
        plot_convergence(dt_vals, ke_err_eul, ke_err_rich)

        # ----- pick a reasonable dt (the smallest that gives <0.5 % error) -----
        good_dt = dt_vals[-1]
        meth_e = Method(0, r0, v0, good_dt, t_end, q, m, E, B, 'euler')
        t_e, r_e, v_e = meth_e.run()
        meth_r = Method(0, r0, v0, good_dt, t_end, q, m, E, B, 'euler-richardson')
        t_r, r_r, v_r = meth_r.run()

        # ----- 3‑D trajectory -----
        plot_3d_trajectory(r_e, r_r,
                           title=f'Pure B‑field – test #{idx}')

        # ----- kinetic‑energy vs time -----
        KE_e = 0.5 * m * np.sum(v_e ** 2, axis=1)
        KE_r = 0.5 * m * np.sum(v_r ** 2, axis=1)
        plot_energy(t_e, KE_e, KE_r,
                    title=f'Kinetic energy (pure B) – test #{idx}')


def run_crossed_test():
    """Task (d): simultaneous E and B fields, different relative orientations."""
    q = -1.0
    m = 1.0
    t_end = 15.0
    dt = 0.01
    r0 = np.zeros(3)

    # ------------------------------------------------------------------
    # (1) E ⟂ B  (E along x, B along z) – classic E×B drift
    # ------------------------------------------------------------------
    E_perp = np.array([0.5, 0.0, 0.0])
    B_perp = np.array([0.0, 0.0, 0.5])

    # drift velocity that cancels the force
    v_drift = drift_velocity(E_perp, B_perp)

    # three initial velocities
    init_vels = [
        np.zeros(3),          # start from rest
        v_drift,              # exactly the drift velocity
        np.array([0.2, 0.1, 0.0])   # arbitrary non‑drift start
    ]

    for idx, v0 in enumerate(init_vels, 1):
        print(f'\n--- Crossed (E⊥B) test #{idx}  v0={v0} ---')
        meth_e = Method(0, r0, v0, dt, t_end, q, m, E_perp, B_perp, 'euler')
        t_e, r_e, v_e = meth_e.run()
        meth_r = Method(0, r0, v0, dt, t_end, q, m, E_perp, B_perp, 'euler-richardson')
        t_r, r_r, v_r = meth_r.run()

        # 3‑D trajectories
        plot_3d_trajectory(r_e, r_r,
                           title=f'Crossed E⊥B – init #{idx}')

        # velocity magnitude (should tend to constant |v_d| after a few cyclotron periods)
        vmag_e = np.linalg.norm(v_e, axis=1)
        vmag_r = np.linalg.norm(v_r, axis=1)
        plot_1d(t_e, vmag_e, vmag_r,
                ylabel='|v|',
                title=f'|v| vs t (E⊥B) – init #{idx}')

    # ------------------------------------------------------------------
    # (2) E ∥ B  (both along z)
    # ------------------------------------------------------------------
    E_par = np.array([0.0, 0.0, 0.4])
    B_par = np.array([0.0, 0.0, 0.5])

    init_vels_par = [
        np.zeros(3),
        np.array([0.0, 0.0, 0.2]),
        np.array([0.3, 0.0, 0.0])
    ]

    for idx, v0 in enumerate(init_vels_par, 1):
        print(f'\n--- Crossed (E∥B) test #{idx}  v0={v0} ---')
        meth_e = Method(0, r0, v0, dt, t_end, q, m, E_par, B_par, 'euler')
        t_e, r_e, v_e = meth_e.run()
        meth_r = Method(0, r0, v0, dt, t_end, q, m, E_par, B_par, 'euler-richardson')
        t_r, r_r, v_r = meth_r.run()

        plot_3d_trajectory(r_e, r_r,
                           title=f'Crossed E∥B – init #{idx}')

        # look at the component parallel to the fields
        plot_1d(t_e, r_e[:, 2], r_r[:, 2],
                ylabel='z(t)',
                title=f'z vs t (E∥B) – init #{idx}')


def run_two_body_test():
    """Task (e): two particles with Coulomb interaction (repulsive & attractive)."""

    # --------------------------------------------------------------------
    # Physical constants (SI)
    # --------------------------------------------------------------------
    qe = -1.602e-19          # electron charge
    me = 9.109e-31           # electron mass
    qp = +1.602e-19          # proton‑like positive charge
    mp = 1.672e-27           # proton mass (optional, can also use me)

    # No external fields for this part
    E_ext = np.zeros(3)
    B_ext = np.zeros(3)

    # Initial geometry
    d0 = 1e-9                # 1 nm separation
    r1_0 = np.array([-d0/2, 0., 0.])
    r2_0 = np.array([ d0/2, 0., 0.])
    v1_0 = np.zeros(3)
    v2_0 = np.zeros(3)

    # Integration parameters – choose a time long enough to see motion,
    # but not so long that the particles fly out of the figure.
    t_end = 2e-12            # 2 ps (adjust if you want a more gradual curve)
    dt    = 1e-16            # small enough for a stable Richardson step

    # --------------------------------------------------------------------
    # (i) Repulsive case: electron–electron
    # --------------------------------------------------------------------
    print('\n--- Repulsive case (e‑e) ---')
    two_rep = TwoParticleMethod(
                0, r1_0, v1_0, r2_0, v2_0,
                dt, t_end,
                qe, me, qe, me,
                E_ext, B_ext,
                method='euler-richardson')
    t, r1, v1, r2, v2 = two_rep.run()

    # distance vs. time
    dist = np.linalg.norm(r1 - r2, axis=1)
    plt.figure()
    plt.plot(t, dist)
    plt.xlabel('time (s)')
    plt.ylabel('distance (m)')
    plt.title('Repulsive e‑e separation')
    # set a y‑limit that actually shows the growth
    plt.ylim(0, dist.max()*1.1)
    plt.grid(True)
    plt.show()

    # total mechanical energy
    KE = 0.5*me*(np.sum(v1**2, axis=1) + np.sum(v2**2, axis=1))
    PE = (qe*qe) / (TwoParticleMethod.epsilon0 * dist)
    E_tot = KE + PE
    plt.figure()
    plt.plot(t, E_tot)
    plt.xlabel('time (s)')
    plt.ylabel('E_tot (J)')
    plt.title('Repulsive e‑e – total energy (should be constant)')
    plt.grid(True)
    plt.show()

    # --------------------------------------------------------------------
    # (ii) Attractive case: electron – proton
    # --------------------------------------------------------------------
    print('\n--- Attractive case (e‑p) ---')
    two_att = TwoParticleMethod(
                0, r1_0, v1_0, r2_0, v2_0,
                dt, t_end,
                qe, me, qp, mp,
                E_ext, B_ext,
                method='euler-richardson')
    t, r1, v1, r2, v2 = two_att.run()

    dist = np.linalg.norm(r1 - r2, axis=1)
    plt.figure()
    plt.plot(t, dist)
    plt.xlabel('time (s)')
    plt.ylabel('distance (m)')
    plt.title('Attractive e‑p separation')
    plt.ylim(0, dist.max()*1.1)
    plt.grid(True)
    plt.show()

    KE = 0.5*me*np.sum(v1**2, axis=1) + 0.5*mp*np.sum(v2**2, axis=1)
    PE = (qe*qp) / (TwoParticleMethod.epsilon0 * dist)   # negative
    E_tot = KE + PE
    plt.figure()
    plt.plot(t, E_tot)
    plt.xlabel('time (s)')
    plt.ylabel('E_tot (J)')
    plt.title('Attractive e‑p – total energy (constant)')
    plt.grid(True)
    plt.show()


# ----------------------------------------------------------------------
# 6️⃣  MAIN DRIVER – uncomment the block you want to run
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Task (b) – electric field validation
    # run_electric_test()

    # Task (c) – magnetic field validation
    # run_magnetic_test()

    # Task (d) – crossed fields
    # run_crossed_test()

    # Task (e) – two‑particle Coulomb interaction
    run_two_body_test()

    print('All test functions are defined.  Uncomment the one you want to execute.')
