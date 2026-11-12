"""File IO: JSON, camera files, STL meshes and a rotation-aware video reader."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("jawtrack.io")


# ----------------------------------------------------------------------------- JSON
def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, default=_np_default, indent=indent)
    log.info("wrote %s", path)


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- camera
def camera_dict(K, dist, width, height, source, **extra):
    d = {"K": np.asarray(K, float).tolist(),
         "dist": [float(x) for x in np.asarray(dist, float).ravel()[:5]] + [0.0] * max(0, 5 - np.asarray(dist).size),
         "width": int(width), "height": int(height), "source": source}
    d.update(extra)
    return d


def load_camera(path_or_dict):
    cam = read_json(path_or_dict) if not isinstance(path_or_dict, dict) else path_or_dict
    if "K" not in cam and isinstance(cam.get("camera"), dict):
        # nested form {"camera": {K, dist, source}, "width", "height"} (raw_tracking / testdata style)
        inner = dict(cam["camera"])
        for k in ("width", "height"):
            if k in cam and k not in inner:
                inner[k] = cam[k]
        cam = inner
    if "K" not in cam:
        raise ValueError("camera file has no 'K'")
    return cam


def adapt_camera(cam: dict, width: int, height: int) -> dict:
    """Return intrinsics valid for an image of (width, height).

    * same size: unchanged
    * same aspect, different size (e.g. 4K calibration, 1080p video): K scaled (warning)
    * swapped orientation (portrait vs landscape): fx/fy and cx/cy swapped, p1/p2 swapped
      (exact for a centred principal point, approximate otherwise) (warning)
    """
    K = np.asarray(cam["K"], float).copy()
    dist = np.asarray(cam.get("dist", [0] * 5), float).copy()
    cw, ch = cam.get("width"), cam.get("height")
    out = dict(cam)
    if cw is None or (cw == width and ch == height):
        return out
    if cw == height and ch == width:
        log.warning("camera calibrated at %dx%d but images are %dx%d: transposing intrinsics "
                    "(approximate; calibrate in the same orientation as the recording)", cw, ch, width, height)
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1]])
        if dist.size >= 4:
            dist[2], dist[3] = dist[3], dist[2]
        cw, ch = ch, cw
    if cw != width or ch != height:
        sx, sy = width / cw, height / ch
        if abs(sx - sy) > 0.01:
            raise ValueError(f"camera {cw}x{ch} incompatible with images {width}x{height} (aspect differs)")
        log.warning("scaling intrinsics from %dx%d to %dx%d", cw, ch, width, height)
        K[0, :] *= sx
        K[1, :] *= sy
        K[2] = [0, 0, 1]
    out.update(K=K.tolist(), dist=dist.tolist(), width=width, height=height)
    return out


# ----------------------------------------------------------------------------- STL
def load_mesh(path):
    import trimesh
    m = trimesh.load_mesh(str(path), process=False)
    if not hasattr(m, "vertices"):
        raise ValueError(f"{path}: not a triangle mesh")
    return m


def save_mesh(mesh, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


# ----------------------------------------------------------------------------- video
_ROT_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def ffprobe_info(path) -> dict:
    """Minimal ffprobe of the first video stream: width, height (coded), fps, rotation (deg, as
    reported by the display matrix: positive = counter-clockwise), codec, nb_frames."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                              "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
        s = json.loads(out.stdout)["streams"][0]
    except Exception as e:  # pragma: no cover
        log.warning("ffprobe failed: %s", e)
        return {}
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if "rotate" in s.get("tags", {}):
        rot = int(s["tags"]["rotate"])
    num, den = (s.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {"width": int(s.get("width", 0)), "height": int(s.get("height", 0)), "fps": fps,
            "rotation": rot, "codec": s.get("codec_name"), "nb_frames": int(s.get("nb_frames", 0) or 0)}


class VideoReader:
    """Iterate display-oriented BGR frames of a video.

    Rotation is handled EXPLICITLY: OpenCV auto-rotation is disabled, the rotation tag is read
    (OpenCV ``CAP_PROP_ORIENTATION_META`` = clockwise degrees to apply, cross-checked with ffprobe,
    whose display-matrix value is counter-clockwise) and frames are rotated here, so the frames
    always match what the user sees on the phone. If OpenCV cannot decode the file (e.g. a HEVC
    build without decoder), frames are piped through ffmpeg (which applies the rotation itself).
    """

    def __init__(self, path, force_ffmpeg: bool = False):
        self.path = str(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self.probe = ffprobe_info(self.path)
        self.backend = None
        self.rotation_cw = 0
        if not force_ffmpeg:
            cap = cv2.VideoCapture(self.path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
                ok, fr = cap.read()
                if ok and fr is not None:
                    meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
                    rot = int(round(meta)) % 360 if meta >= 0 else (-self.probe.get("rotation", 0)) % 360
                    if self.probe and (-self.probe.get("rotation", 0)) % 360 != rot:
                        log.warning("rotation tag mismatch OpenCV=%s ffprobe=%s; using ffprobe",
                                    rot, self.probe.get("rotation"))
                        rot = (-self.probe.get("rotation", 0)) % 360
                    self.rotation_cw = rot
                    self.fps = cap.get(cv2.CAP_PROP_FPS) or self.probe.get("fps", 30.0)
                    self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.probe.get("nb_frames", 0)
                    fr = self._rot(fr)
                    self.height, self.width = fr.shape[:2]
                    self.backend = "opencv"
                cap.release()
        if self.backend is None:
            if not shutil.which("ffmpeg"):
                raise RuntimeError(f"cannot decode {self.path} with OpenCV and ffmpeg is not available")
            log.warning("OpenCV cannot decode %s -> decoding through an ffmpeg pipe", self.path)
            p = self.probe
            if not p:
                raise RuntimeError(f"ffprobe could not read {self.path}")
            w, h = p["width"], p["height"]
            if p["rotation"] % 180 != 0:
                w, h = h, w
            self.width, self.height, self.fps = w, h, p["fps"] or 30.0
            self.n_frames = p["nb_frames"]
            self.backend = "ffmpeg"
        log.info("video %s: %dx%d (display orientation), %.3f fps, ~%d frames, rotation tag %d deg cw, "
                 "backend %s, codec %s", self.path, self.width, self.height, self.fps, self.n_frames,
                 self.rotation_cw, self.backend, self.probe.get("codec"))

    def _rot(self, fr):
        code = _ROT_CODES.get(self.rotation_cw)
        return cv2.rotate(fr, code) if code is not None else fr

    def frames(self, every: int = 1, max_frames: int | None = None, start: int = 0):
        """Yield (index, t_seconds, frame) for every ``every``-th frame."""
        every = max(1, int(every))
        n_out = 0
        if self.backend == "opencv":
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            i = 0
            while True:
                if i < start or (i - start) % every:
                    if not cap.grab():
                        break
                    i += 1
                    continue
                ok, fr = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not np.isfinite(t) or t <= 0 and i > 0:
                    t = i / self.fps
                yield i, float(i / self.fps if t <= 0 else t), self._rot(fr)
                n_out += 1
                i += 1
                if max_frames and n_out >= max_frames:
                    break
            cap.release()
        else:
            cmd = ["ffmpeg", "-v", "error", "-i", self.path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            nbytes = self.width * self.height * 3
            i = 0
            try:
                while True:
                    buf = proc.stdout.read(nbytes)
                    if len(buf) < nbytes:
                        break
                    if i >= start and (i - start) % every == 0:
                        fr = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
                        yield i, i / self.fps, fr.copy()
                        n_out += 1
                        if max_frames and n_out >= max_frames:
                            break
                    i += 1
            finally:
                proc.kill()

    def read_frame(self, index: int):
        """Random access to one frame (sequential decode for the ffmpeg backend)."""
        if self.backend == "opencv":
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, fr = cap.read()
            cap.release()
            if ok:
                return self._rot(fr)
        for i, _, fr in self.frames(start=index, max_frames=1):
            return fr
        return None


def read_image(path):
    """Read a still image (EXIF orientation applied by OpenCV's imread)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path} (HEIC? convert to JPG/PNG first)")
    return img
