"""Physiological ground-truth mandibular kinematics T_S(t) (lower MI -> pose at t, upper frame).

Model: two condyle points (Bonwill positions) move along prescribed paths; the mandible is the
rigid body that (1) carries the intercondylar axis from its MI direction to the current condyle
positions by a minimal rotation + midpoint translation and (2) rotates by phi about the current
intercondylar axis (positive phi = opening). phi = max(designed opening, smallest phi >= 0 that
keeps the lower teeth out of the upper teeth), so protrusion/laterotrusion are tooth-guided.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .dentition import heightfield


def min_jerk(t):
    t = np.clip(t, 0, 1)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def rot_axis(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def min_rot(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    return rot_axis(v, np.arctan2(s, a @ b))


# ----------------------------------------------------------------------------- timeline
def build_timeline(kind="full", fps=60.0):
    """Returns (segments, events). events: list of (t0, t1, type, params)."""
    ev = []
    segs = []
    t = 0.0

    def hold(dur):
        nonlocal t
        t += dur

    def reps(label, n, out, hold_, back, rest, amp_list):
        nonlocal t
        t0 = t
        for k in range(n):
            ev.append((t, t + out + hold_ + back, label, dict(out=out, hold=hold_, back=back, amp=amp_list[k])))
            t += out + hold_ + back + rest
        segs.append(dict(label=label, t0=round(t0, 4), t1=round(t - rest, 4)))

    if kind == "full":
        segs.append(dict(label="MI", t0=0.0, t1=3.0)); hold(3.0)
        reps("open_close", 3, 0.85, 0.15, 0.8, 0.15, [0.96, 1.0, 0.98]); hold(0.25)
        reps("protrusion", 3, 0.65, 0.15, 0.6, 0.15, [0.97, 1.0, 0.99]); hold(0.25)
        reps("laterotrusion_right", 3, 0.6, 0.12, 0.55, 0.15, [0.97, 1.0, 0.98]); hold(0.25)
        reps("laterotrusion_left", 3, 0.6, 0.12, 0.55, 0.15, [0.98, 1.0, 0.97]); hold(0.25)
        t0 = t
        ev.append((t, t + 5.0, "chewing", dict(cycles=6))); t += 5.0
        segs.append(dict(label="chewing", t0=round(t0, 4), t1=round(t, 4)))
        segs.append(dict(label="MI", t0=round(t, 4), t1=round(t + 3.0, 4))); hold(3.0)
    elif kind == "quick":
        segs.append(dict(label="MI", t0=0.0, t1=2.0)); hold(2.0)
        reps("open_close", 2, 0.85, 0.15, 0.8, 0.15, [0.97, 1.0]); hold(0.2)
        reps("protrusion", 2, 0.65, 0.15, 0.6, 0.15, [0.98, 1.0]); hold(0.2)
        rest = 10.0 - t
        segs.append(dict(label="MI", t0=round(t, 4), t1=10.0)); hold(rest)
    else:
        raise ValueError(kind)
    return segs, ev, t


def excursion_u(t, e):
    t0, t1, typ, p = e
    tau = t - t0
    if tau <= 0 or t >= t1:
        return 0.0
    if tau < p["out"]:
        return p["amp"] * min_jerk(tau / p["out"])
    tau -= p["out"]
    if tau < p["hold"]:
        return p["amp"]
    tau -= p["hold"]
    return p["amp"] * (1 - min_jerk(tau / p["back"]))


def chew_state(t, e):
    """Chewing on the RIGHT side: returns (open_u, lat_u) in [0,1]."""
    t0, t1, typ, p = e
    if t <= t0 or t >= t1:
        return 0.0, 0.0
    ph = (t - t0) / (t1 - t0) * p["cycles"]
    k = int(ph)
    x = ph - k
    amp = [0.9, 1.0, 0.95, 1.05, 0.92, 1.0][k % 6]
    # opening 0..0.42, closing 0.42..0.85, occlusal phase (contact glide) 0.85..1
    if x < 0.42:
        o = min_jerk(x / 0.42)
    elif x < 0.85:
        o = 1 - min_jerk((x - 0.42) / 0.43)
    else:
        o = 0.0
    # lateral deviation to the working (right) side mostly during closing
    if x < 0.42:
        lat = 0.35 * min_jerk(x / 0.42)
    elif x < 0.7:
        lat = 0.35 + 0.65 * min_jerk((x - 0.42) / 0.28)
    else:
        lat = 1 - min_jerk((x - 0.7) / 0.3)
    return amp * o, amp * lat


# ----------------------------------------------------------------------------- model
class JawModel:
    def __init__(self, lm, upper_mesh, lower_mesh, theta_L=35.0, theta_R=38.0, bennett_L=10.0,
                 bennett_R=8.0, curv=0.012, side_shift=0.06):
        fr = lm["frame"]
        self.n, self.f, self.l = fr["n"], fr["f"], fr["l"]
        self.cL, self.cR = lm["condyle_left"], lm["condyle_right"]
        self.ip = lm["incisal_point"]
        self.curv = curv
        # nominal path angle chosen so that the CHORD at ~5.5 mm condylar travel ~= target
        self.thL = np.radians(theta_L) + np.arctan(curv * 5.5)
        self.thR = np.radians(theta_R) + np.arctan(curv * 5.5)
        self.bL, self.bR = np.radians(bennett_L), np.radians(bennett_R)
        self.side_shift = side_shift
        self.a0 = self.cL - self.cR
        self.m0 = 0.5 * (self.cL + self.cR)
        # contact model: upper min-y heightfield + lower crown sample points
        self.H = heightfield([upper_mesh], lo=True, res=0.2, samples=3)
        v = lower_mesh.vertices
        sel = v[:, 1] > np.percentile(v[:, 1], 55)
        pts = v[sel]
        rng = np.random.default_rng(1)
        if len(pts) > 12000:
            pts = pts[rng.choice(len(pts), 12000, replace=False)]
        self.P = pts
        self.pen0 = self._maxpen(np.eye(4))

    # paths ------------------------------------------------------------------------------
    def sag_path(self, s, th):
        f, n = self.f, self.n
        d = np.array([np.cos(th), -np.sin(th)])
        perp = np.array([np.sin(th), np.cos(th)])  # upward-ish: path flattens with travel
        fx, nx = s * d + self.curv * s * s * perp
        return fx * f + nx * n

    def orbit_path(self, s, th, B, medial):
        p = self.sag_path(s, th)
        return p + np.tan(B) * (p @ self.f) * medial

    def pose(self, dL, dR, phi, anchor="M"):
        """anchor: which point is placed exactly ('L','R' condyle or 'M' midpoint). In
        laterotrusion the ORBITING condyle is the anchor so its path is exactly as designed
        (the working condyle absorbs the tiny inter-condylar distance change)."""
        cL, cR = self.cL + dL, self.cR + dR
        a1 = cL - cR
        R1 = min_rot(self.a0, a1)
        R = rot_axis(a1, phi) @ R1
        src, dst = {"M": (self.m0, 0.5 * (cL + cR)), "L": (self.cL, cL), "R": (self.cR, cR)}[anchor]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = dst - R @ src
        return T

    # contacts ---------------------------------------------------------------------------
    def _maxpen(self, T):
        p = self.P @ T[:3, :3].T + T[:3, 3]
        H = self.H
        hf = H["hf"]
        fx = (p[:, 0] - H["x0"]) / H["res"]
        fz = (p[:, 2] - H["z0"]) / H["res"]
        ix = np.floor(fx).astype(int)
        iz = np.floor(fz).astype(int)
        ok = (ix >= 0) & (iz >= 0) & (ix < hf.shape[1] - 1) & (iz < hf.shape[0] - 1)
        if not ok.any():
            return -np.inf
        ix, iz, y = ix[ok], iz[ok], p[ok, 1]
        h = np.minimum(np.minimum(hf[iz, ix], hf[iz, ix + 1]), np.minimum(hf[iz + 1, ix], hf[iz + 1, ix + 1]))
        pen = y - h
        pen = pen[np.isfinite(pen)]
        return pen.max() if len(pen) else -np.inf

    def phi_contact(self, dL, dR, anchor="M", lo=0.0, hi=np.radians(30)):
        tol = self.pen0
        if self._maxpen(self.pose(dL, dR, lo, anchor)) <= tol:
            return lo
        a, b = lo, hi
        for _ in range(16):
            c = 0.5 * (a + b)
            if self._maxpen(self.pose(dL, dR, c, anchor)) <= tol:
                b = c
            else:
                a = c
        return b

    # full state -> displacements ---------------------------------------------------------
    def displacements(self, kind, u, amps):
        z = np.zeros(3)
        if kind == "open_close":
            s = amps["open_s"] * smoothstep((u - 0.4) / 0.6)
            return self.sag_path(s, self.thL), self.sag_path(s, self.thR), amps["open_phi"] * u, "M"
        if kind == "protrusion":
            s = amps["prot_s"] * u
            return self.sag_path(s, self.thL), self.sag_path(s, self.thR), 0.0, "M"
        if kind in ("laterotrusion_right", "laterotrusion_left"):
            if kind == "laterotrusion_right":
                s = amps["lat_s_R"] * u  # left condyle orbits (medial = -l), right works
                dL = self.orbit_path(s, self.thL, self.bL, -self.l)
                dR = -self.l * self.side_shift * s
                return dL, dR, 0.0, "L"
            s = amps["lat_s_L"] * u
            dR = self.orbit_path(s, self.thR, self.bR, self.l)
            dL = self.l * self.side_shift * s
            return dL, dR, 0.0, "R"
        return z, z, 0.0, "M"

    def tune(self, target_open=45.0, target_prot=9.0, target_lat=10.0):
        """Find amplitudes giving the target incisal excursions (geometry only, phi_contact=0)."""
        amps = dict(open_s=12.0, open_phi=0.0, prot_s=9.0, lat_s_R=10.0, lat_s_L=10.0)

        def ipd(T):
            return T[:3, :3] @ self.ip + T[:3, 3] - self.ip

        def bis(fn, lo, hi, target):
            for _ in range(50):
                c = 0.5 * (lo + hi)
                if fn(c) < target:
                    lo = c
                else:
                    hi = c
            return 0.5 * (lo + hi)

        def open_amt(phi):
            a = dict(amps, open_phi=phi)
            dL, dR, ph, an = self.displacements("open_close", 1.0, a)
            return np.linalg.norm(ipd(self.pose(dL, dR, ph, an)))

        amps["open_phi"] = bis(open_amt, 0.0, 0.9, target_open)

        def prot_amt(s):
            a = dict(amps, prot_s=s)
            dL, dR, ph, an = self.displacements("protrusion", 1.0, a)
            return ipd(self.pose(dL, dR, ph, an)) @ self.f

        amps["prot_s"] = bis(prot_amt, 0.0, 20.0, target_prot)

        def lat_amt(s, side):
            kind = "laterotrusion_right" if side == "R" else "laterotrusion_left"
            a = dict(amps, **{"lat_s_" + side: s})
            dL, dR, ph, an = self.displacements(kind, 1.0, a)
            d = ipd(self.pose(dL, dR, ph, an)) @ self.l
            return -d if side == "R" else d

        for side in "RL":
            amps["lat_s_" + side] = bis(lambda s: lat_amt(s, side), 0.0, 25.0, target_lat)
        # chewing: ~18 mm opening, ~5 mm lateral deviation at the incisors
        amps["chew_phi"] = amps["open_phi"] * 0.42
        amps["chew_lat_s"] = amps["lat_s_R"] * 0.5
        self.amps = amps
        return amps

    def generate(self, segs, events, duration, fps=60.0, verbose=True):
        n = int(round(duration * fps))
        times = np.arange(n) / fps
        Ts = np.tile(np.eye(4), (n, 1, 1))
        phis = np.zeros(n)
        mi = np.zeros(n, bool)
        state = []
        for i, t in enumerate(times):
            st = ("MI", 0.0, 0.0)
            for e in events:
                if e[0] < t < e[1]:
                    if e[2] == "chewing":
                        o, la = chew_state(t, e)
                        st = ("chewing", o, la)
                    else:
                        st = (e[2], excursion_u(t, e), 0.0)
            state.append(st)
        A = self.amps
        disp = []
        for i, (kind, u, la) in enumerate(state):
            if kind == "chewing":
                a = dict(A, lat_s_R=A["chew_lat_s"])
                dL, dR, _, an = self.displacements("laterotrusion_right", la, a)
                ph = A["chew_phi"] * u
            else:
                dL, dR, ph, an = self.displacements(kind, u, A)
            disp.append((dL, dR, ph, an))
            if kind == "MI" or (u == 0.0 and la == 0.0):
                mi[i] = True
        # contact solve
        for i, (dL, dR, ph, an) in enumerate(disp):
            if mi[i]:
                continue
            pc = self.phi_contact(dL, dR, an)
            phis[i] = max(ph, pc)
            if verbose and i % 300 == 0:
                print(f"    kinematics frame {i}/{n}")
        # light smoothing of phi (keeps the max with the contact constraint)
        sm = gaussian_filter1d(phis, 1.5)
        phis = np.maximum(sm, phis)
        phis[mi] = 0.0
        for i, (dL, dR, _, an) in enumerate(disp):
            if not mi[i]:
                Ts[i] = self.pose(dL, dR, phis[i], an)
        return times, Ts, phis, state
