#!/usr/bin/env python3
"""Generate a simple viewer test fixture (NOT anatomical, NOT the synth dataset).

Writes to viewer/dev/fixture/:
  upper.stl, lower.stl (binary), lower_ascii.stl (ASCII copy of lower),
  motion.json (SPEC 5.2), ground_truth.json (same motion, "true" params),
  registration_landmarks.json (SPEC 4), reg_frame.png (1920x1080, coloured dots
  at the img_px of the landmarks), video.webm + video.mp4 (counter clip).

STL frame per SPEC 6: x = patient LEFT, y = SUPERIOR, z = ANTERIOR, origin ~ upper incisal point.
"""
import argparse, json, math, os, struct, subprocess
import numpy as np

RES = 1.0

def superellipsoid(c, r, n=18, m=12, e=0.35, flat_bottom=None):
    n = int(n * RES); m = int(m * RES)
    """Rounded-block mesh centred at c with radii r. Returns (V, F)."""
    us = np.linspace(-math.pi / 2, math.pi / 2, m + 1)
    vs = np.linspace(-math.pi, math.pi, n, endpoint=False)
    sgn = lambda x, p: np.sign(x) * np.abs(x) ** p
    V = []
    for u in us:
        for v in vs:
            x = sgn(math.cos(u), e) * sgn(math.cos(v), e)
            z = sgn(math.cos(u), e) * sgn(math.sin(v), e)
            y = sgn(math.sin(u), e)
            V.append([c[0] + r[0] * x, c[1] + r[1] * y, c[2] + r[2] * z])
    V = np.array(V)
    F = []
    for i in range(m):
        for j in range(n):
            a = i * n + j; b = i * n + (j + 1) % n
            c2 = (i + 1) * n + j; d = (i + 1) * n + (j + 1) % n
            F.append([a, c2, b]); F.append([b, c2, d])
    return V, np.array(F)

def rotY(V, c, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    P = V - c
    return np.c_[P[:, 0] * ca + P[:, 2] * sa, P[:, 1], -P[:, 0] * sa + P[:, 2] * ca] + c

def arch(upper):
    """Teeth along a parabolic arch. Upper crowns hang down to y=0 (incisors overlap)."""
    Vs, Fs, off = [], [], 0
    half = 7
    for side in (-1, 1):
        for k in range(half):
            s = (k + 0.5) / half
            x = side * (4 + 23 * s ** 0.8) * (1.0 if upper else 0.94)
            z = -48 * s ** 1.6 - (0 if upper else 1.8)
            w = 3.6 + 2.6 * s ** 1.5            # crown half-width grows posteriorly
            if upper:
                ylo = -2.0 if k < 2 else 0.0     # 2 mm overbite on incisors
                c = [x, ylo + 4.5, z]
            else:
                ytop = -0.5 if k < 2 else 0.0
                c = [x, ytop - 4.5, z]
            V, F = superellipsoid(c, [w * 0.9, 4.5, 3.4 + 1.6 * s], e=0.45 if k < 2 else 0.3)
            ang = math.atan2(2 * 48 * 1.6 * s ** 0.6 / 23, 1) * side * 0.6
            V = rotY(V, np.array(c), ang)
            Vs.append(V); Fs.append(F + off); off += len(V)
    # gingiva/base slab: chain of flat blobs
    for side in (-1, 1):
        for k in range(9):
            s = (k + 0.5) / 9
            x = side * (4 + 23 * s ** 0.8) * (1.0 if upper else 0.94)
            z = -48 * s ** 1.6 - (0 if upper else 1.8)
            c = [x, 11.5 if upper else -11.5, z]
            V, F = superellipsoid(c, [7.5, 3.5, 7.5], n=16, m=8, e=0.6)
            Vs.append(V); Fs.append(F + off); off += len(V)
    V = np.vstack(Vs); F = np.vstack(Fs)
    if not upper:  # flip winding consistency is fine (superellipsoid outward), keep as is
        pass
    return V, F

def write_stl_bin(path, V, F):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    with open(path, 'wb') as f:
        f.write(b'fixture'.ljust(80, b' ')); f.write(struct.pack('<I', len(F)))
        rec = np.zeros(len(F), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
        rec['n'] = n; rec['v'] = tri
        f.write(rec.tobytes())

def write_stl_ascii(path, V, F):
    tri = V[F]
    with open(path, 'w') as f:
        f.write('solid fixture\n')
        for t in tri:
            n = np.cross(t[1] - t[0], t[2] - t[0]); n /= np.linalg.norm(n) + 1e-12
            f.write(f' facet normal {n[0]:e} {n[1]:e} {n[2]:e}\n  outer loop\n')
            for v in t: f.write(f'   vertex {v[0]:e} {v[1]:e} {v[2]:e}\n')
            f.write('  endloop\n endfacet\n')
        f.write('endsolid fixture\n')

def rot_axis(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

def about(R, p, t=np.zeros(3)):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p - R @ p + t; return T

CL = np.array([52.5, 36.0, -91.0]); CR = np.array([-52.5, 36.0, -91.0])
HING = (CL + CR) / 2
EMIN = np.array([0, -math.sin(math.radians(35)), math.cos(math.radians(35))])

def pose_open(a):  # a in 0..1
    ang = math.radians(28) * a
    s = 12.0 * max(0.0, a - 0.25) / 0.75
    return about(rot_axis([1, 0, 0], ang), HING, EMIN * s)

def pose_protr(a):
    return about(rot_axis([1, 0, 0], math.radians(3) * a), HING, EMIN * 9.0 * a)

def pose_lat(a, right=True):
    work = CR if right else CL
    beta = math.radians(35)
    d = np.array([0, math.cos(beta), math.sin(beta)])
    sign = -1 if right else 1
    R = rot_axis(d, sign * math.radians(9) * a)
    side = np.array([-0.8 if right else 0.8, 0, 0]) * min(1, a * 3)
    return about(R, work, side)

def build_motion(fps=60.0):
    seq = []  # (label, duration, func(u)->T)
    I = lambda u: np.eye(4)
    bump = lambda u, n: 0.5 - 0.5 * math.cos(2 * math.pi * n * u)
    seq.append(('MI', 2.0, I))
    seq.append(('open_close', 9.0, lambda u: pose_open(bump(u, 3))))
    seq.append(('protrusion', 6.0, lambda u: pose_protr(bump(u, 2))))
    seq.append(('laterotrusion_right', 5.0, lambda u: pose_lat(bump(u, 2), True)))
    seq.append(('laterotrusion_left', 5.0, lambda u: pose_lat(bump(u, 2), False)))
    def chew(u):
        ph = 2 * math.pi * 6 * u
        a = 0.5 - 0.5 * math.cos(ph)
        T1 = pose_open(0.35 * a); T2 = pose_lat(0.5 * a * (0.5 + 0.5 * math.sin(ph)), True)
        return T1 @ T2
    seq.append(('chewing', 5.0, chew))
    seq.append(('MI', 2.0, I))
    frames, segs, t0 = [], [], 0.0
    for label, dur, fn in seq:
        segs.append({'label': label, 't0': round(t0, 4), 't1': round(t0 + dur, 4)})
        n = int(round(dur * fps))
        for k in range(n):
            frames.append((t0 + k / fps, fn(k / n)))
        t0 += dur
    return frames, segs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixture'))
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--res', type=float, default=1.0, help='mesh resolution multiplier (e.g. 14 -> ~2M triangles/jaw, ~100 MB binary STL)')
    a = ap.parse_args()
    global RES; RES = a.res
    os.makedirs(a.out, exist_ok=True)
    Vu, Fu = arch(True); Vl, Fl = arch(False)
    write_stl_bin(os.path.join(a.out, 'upper.stl'), Vu, Fu)
    write_stl_bin(os.path.join(a.out, 'lower.stl'), Vl, Fl)
    if a.res <= 2: write_stl_ascii(os.path.join(a.out, 'lower_ascii.stl'), Vl, Fl)
    fps = 60.0
    frames, segs = build_motion(fps)
    incisal = np.array([0.0, -0.5, -2.2])
    rng = np.random.default_rng(1)
    out_frames, gt_frames = [], []
    for i, (t, T) in enumerate(frames):
        valid = not (14.0 <= t < 14.4 or 26.0 <= t < 26.15)
        Tn = T.copy(); Tn[:3, 3] += rng.normal(0, 0.03, 3)
        out_frames.append({'t': round(t, 5), 'T': [float(x) for x in Tn.reshape(-1)] if valid else None,
                           'valid': valid, 'quality': round(float(0.75 + 0.25 * rng.random()), 3) if valid else 0.0})
        gt_frames.append({'t': round(t, 5), 'T': [float(x) for x in T.reshape(-1)]})
    lm = {'incisal_point': incisal.tolist(), 'condyle_left': CL.tolist(), 'condyle_right': CR.tolist(),
          'occlusal_plane': {'point': [0, 0, -20], 'normal': [0, 1, 0]}, 'condyle_method': 'average_bonwill'}
    true_params = {'max_opening_mm': 50.6, 'max_protrusion_mm': 9.1, 'max_laterotrusion_right_mm': 9.4,
                   'max_laterotrusion_left_mm': 9.4, 'sagittal_condylar_inclination_left_deg': 35.0,
                   'sagittal_condylar_inclination_right_deg': 35.0, 'bennett_angle_left_deg': 9.0,
                   'bennett_angle_right_deg': 9.0, 'incisal_guidance_deg': 48.0}
    est = {k: round(v * (1 + rng.normal(0, 0.06)), 1) for k, v in true_params.items()}
    est['kinematic_hinge_axis'] = {'point': HING.tolist(), 'direction': [1, 0, 0], 'fit_rms_mm': 0.42}
    est['notes'] = ['Vinkler er målt i forhold til okklusionsplanet (Frankfort ikke tilgængeligt).',
                    'Kondylpunkter er gennemsnitsværdier (Bonwill), ikke målte.',
                    'FIXTURE: syntetiske testdata til viewer-test, ikke en rigtig patient.']
    motion = {'version': 1, 'units': 'mm', 'fps': fps,
              'frame_of_reference': 'upper STL coordinates; T maps lower STL (as exported in MI) to its pose at time t',
              'source': {'video': 'video.mp4', 'video_offset_s': 1.0},
              'frames': out_frames, 'segments': segs, 'landmarks_mm': lm, 'parameters': est,
              'quality': {'tracked_fraction': round(sum(f['valid'] for f in out_frames) / len(out_frames), 4),
                          'registration_rms_px': 1.8, 'jitter_mm_rms_in_MI': 0.05}}
    json.dump(motion, open(os.path.join(a.out, 'motion.json'), 'w'))
    json.dump({'version': 1, 'description': 'fixture ground truth', 'fps': fps, 'frames': gt_frames,
               'segments': segs, 'landmarks_mm': lm, 'parameters': true_params},
              open(os.path.join(a.out, 'ground_truth.json'), 'w'))
    # registration still: dots at known pixels
    W, H = 1920, 1080
    pts = [('11_incisal', [868.0, 610.0], [-4.3, -2.0, -0.1], 'upper'),
           ('21_incisal', [1052.0, 612.0], [4.3, -2.0, -0.1], 'upper'),
           ('13_cusp', [742.0, 590.0], [-14.8, 0.2, -8.0], 'upper'),
           ('23_cusp', [1181.0, 588.0], [14.8, 0.2, -8.0], 'upper'),
           ('16_mb', [610.0, 560.0], [-24.0, 0.3, -30.0], 'upper'),
           ('26_mb', [1311.0, 561.0], [24.0, 0.3, -30.0], 'upper')]
    from PIL import Image, ImageDraw
    im = Image.new('RGB', (W, H), (38, 32, 34)); d = ImageDraw.Draw(im)
    d.ellipse([560, 420, 1360, 820], fill=(150, 70, 80))
    d.rectangle([600, 540, 1320, 620], fill=(235, 228, 214))
    for nm, (u, v), _, _ in pts:
        d.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(255, 0, 0))
    im.save(os.path.join(a.out, 'reg_frame.png'))
    reg = {'version': 1, 'image': {'file': 'reg_frame.png', 'frame_index': 120, 'width': W, 'height': H},
           'points': [{'name': n, 'img_px': p, 'stl_mm': s, 'jaw': j} for n, p, s, j in pts],
           'incisal_point_mm': incisal.tolist(), 'occlusal_plane_mm': [incisal.tolist(), [21.0, 0.0, -36.0], [-21.0, 0.0, -36.0]]}
    json.dump(reg, open(os.path.join(a.out, 'registration_landmarks.json'), 'w'), indent=1)
    if not a.no_video:
        dur = frames[-1][0] + 1.0 + 1.0
        for ext, codec in (('webm', ['-c:v', 'libvpx-vp9', '-b:v', '300k']), ('mp4', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p'])):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', f'testsrc=size=480x270:rate=30:duration={dur:.2f}',
                            *codec, os.path.join(a.out, f'video.{ext}')], check=True)
    print('fixture written to', a.out, 'frames', len(out_frames), 'tris', len(Fu), len(Fl))

if __name__ == '__main__':
    main()
