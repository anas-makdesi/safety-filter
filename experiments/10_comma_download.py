"""Download the few arrays we need from comma2k19 (real highway driving, MIT licence).

comma2k19 (Schafer et al. 2018, https://github.com/commaai/comma2k19) is ~100 GB, mostly
video. We need only, per one-minute segment: CAN speed and steering-wheel angle, the
gyroscope, and the camera poses. The zip files on Hugging Face allow byte-range reads,
so only those arrays (about 0.3 MB per segment) are fetched.

Chunks 1-2 are the Toyota RAV4, chunks 3-10 the Honda Civic.

Run: python experiments/10_comma_download.py [--chunks 1 2]
Writes data/comma2k19_raw/<chunk>/<route>__<segment>.npz (not in git).
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_{}.zip"
FILES = {
    "speed_t": "processed_log/CAN/speed/t", "speed": "processed_log/CAN/speed/value",
    "steer_t": "processed_log/CAN/steering_angle/t",
    "steer": "processed_log/CAN/steering_angle/value",
    "gyro_t": "processed_log/IMU/gyro/t", "gyro": "processed_log/IMU/gyro/value",
    "pose_t": "global_pose/frame_times", "pose_q": "global_pose/frame_orientations",
    "pose_v": "global_pose/frame_velocities",
}


def segments(z):
    names = set(z.namelist())
    segs = sorted({n.split("/processed_log/")[0] for n in names if "/processed_log/" in n})
    return [s for s in segs if all(f"{s}/{f}" in names for f in FILES.values())]


def fetch(url, seg, out, tries: int = 4):
    if out.exists():
        return
    for k in range(tries):
        try:
            with RemoteZip(url) as z:
                arrs = {k: np.load(io.BytesIO(z.read(f"{seg}/{f}"))) for k, f in FILES.items()}
            break
        except Exception:          # connection reset by the server: wait and retry
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.rename(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    for c in a.chunks:
        url = URL.format(c)
        d = ROOT / "data" / "comma2k19_raw" / f"chunk{c}"
        d.mkdir(parents=True, exist_ok=True)
        with RemoteZip(url) as z:
            segs = segments(z)
        print(f"chunk {c}: {len(segs)} segments", flush=True)
        jobs = [(url, s, d / (s.split("/", 1)[1].replace("|", "__").replace("/", "__") + ".npz"))
                for s in segs]
        with ThreadPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(lambda j: fetch(*j), jobs)):
                if i % 25 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
