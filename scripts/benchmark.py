"""
benchmark.py
------------
Ayni video uzerinde 3 farkli formati (PT / ONNX / TensorRT Engine)
calistirir ve performans metriklerini karsilastirir.

Olculen metrikler:
    - Ortalama / median / p95 inference suresi (ms)
    - FPS
    - Model dosya boyutu (MB)
    - CPU kullanimi (%) ort/max
    - RAM kullanimi (MB) ort/max
    - GPU utilization (%) ort/max  — NVIDIA GPU gerekli
    - GPU VRAM kullanimi (MB) ort/max
    - Toplam detection sayisi + ortalama confidence
    - Accuracy parity: PT'ye gore frame bazli uyusma orani

ONNX GPU FIX:
    onnxruntime-gpu ile CUDA 13 (cu130) uyumsuzlugu varsa ONNX CPU'da calisir.
    requirements.txt'teki talimatlara gore onnxruntime-gpu'yu yeniden kur.

Kullanim:
    python benchmark.py --video input.mp4 --models_dir models --save_output
"""

import argparse
import os
import time
import json
import threading
from pathlib import Path

# ── ONNX GPU DLL preload fix (ORT >= 1.21) ──────────────────────────────────
# torch import'u ONNX Runtime oturum acilmadan ONCE yapilmali.
# Bu, CUDA/cuDNN DLL'lerinin ORT tarafindan bulunmasini saglar.
try:
    import torch as _torch  # noqa: F401 — sadece DLL preload icin
except ImportError:
    pass
# ────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    GPU_HANDLE = None


class ResourceMonitor:
    """200 ms aralikla CPU / RAM / GPU kullanimini ornekleyen thread."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.running = False
        self.cpu_samples, self.ram_samples = [], []
        self.gpu_util_samples, self.gpu_mem_samples = [], []
        self._thread = None
        self._proc = psutil.Process(os.getpid())

    def _loop(self):
        self._proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        while self.running:
            time.sleep(self.interval)
            try:
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                self.ram_samples.append(self._proc.memory_info().rss / (1024 * 1024))
                if GPU_AVAILABLE and GPU_HANDLE is not None:
                    u = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
                    m = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
                    self.gpu_util_samples.append(u.gpu)
                    self.gpu_mem_samples.append(m.used / (1024 * 1024))
            except Exception:
                pass

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        def _s(lst):
            return round(float(np.mean(lst)), 2) if lst else None

        def _mx(lst):
            return round(float(np.max(lst)), 2) if lst else None

        return {
            "cpu_avg_pct": _s(self.cpu_samples) or 0.0,
            "cpu_max_pct": _mx(self.cpu_samples) or 0.0,
            "ram_avg_mb": _s(self.ram_samples) or 0.0,
            "ram_max_mb": _mx(self.ram_samples) or 0.0,
            "gpu_util_avg_pct": _s(self.gpu_util_samples),
            "gpu_util_max_pct": _mx(self.gpu_util_samples),
            "gpu_mem_avg_mb": _s(self.gpu_mem_samples),
            "gpu_mem_max_mb": _mx(self.gpu_mem_samples),
        }


def _mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def benchmark_model(model_path: str, video_path: str, save_output: bool,
                    output_dir: str, conf: float) -> dict:
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_path}")
    print(f"{'='*60}")

    model = YOLO(model_path)
    model_size = _mb(model_path)
    fmt = Path(model_path).suffix.lstrip(".")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video acilamadi: {video_path}")

    fps_v = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_p = os.path.join(output_dir, f"output_{fmt}.mp4")
        writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))

    # Warm-up
    print("Isinma (warm-up)...")
    ret, frame = cap.read()
    if ret:
        for _ in range(5):
            _ = model(frame, verbose=False, conf=conf)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    inf_times, confidences, per_frame_det = [], [], []
    total_det = 0

    monitor = ResourceMonitor()
    monitor.start()
    t_start = time.time()
    fc = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model(frame, verbose=False, conf=conf)
        inf_times.append((time.time() - t0) * 1000)

        r = results[0]
        nd = len(r.boxes) if r.boxes is not None else 0
        total_det += nd
        per_frame_det.append(nd)
        if nd > 0 and r.boxes.conf is not None:
            confidences.extend(r.boxes.conf.cpu().numpy().tolist())

        if writer is not None:
            writer.write(r.plot())

        fc += 1
        if fc % 30 == 0:
            print(f"  Frame {fc}/{total} | {inf_times[-1]:.1f} ms | det: {nd}")

    total_time = time.time() - t_start
    res = monitor.stop()
    cap.release()
    if writer:
        writer.release()

    avg = float(np.mean(inf_times))
    fps = 1000.0 / avg if avg > 0 else 0

    metrics = {
        "model": model_path,
        "format": fmt,
        "model_size_mb": round(model_size, 2),
        "total_frames": fc,
        "total_time_sec": round(total_time, 2),
        "avg_inference_ms": round(avg, 2),
        "median_inference_ms": round(float(np.median(inf_times)), 2),
        "p95_inference_ms": round(float(np.percentile(inf_times, 95)), 2),
        "min_inference_ms": round(float(np.min(inf_times)), 2),
        "max_inference_ms": round(float(np.max(inf_times)), 2),
        "fps": round(fps, 2),
        "total_detections": total_det,
        "avg_detections_per_frame": round(total_det / max(fc, 1), 2),
        "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "per_frame_detections": per_frame_det,
        **res,
    }

    print(f"\n--- SONUC ({fmt.upper()}) ---")
    for k, v in metrics.items():
        if k in ("per_frame_detections", "model"):
            continue
        print(f"  {k:<25}: {v}")

    return metrics


def compute_parity(results: list) -> dict:
    pt = next((r for r in results if r["format"] == "pt"), None)
    if not pt:
        return {}
    parity = {}
    for r in results:
        if r["format"] == "pt":
            continue
        pc, rc = pt["per_frame_detections"], r["per_frame_detections"]
        n = min(len(pc), len(rc))
        if n == 0:
            continue
        same = sum(1 for i in range(n) if pc[i] == rc[i])
        diff = sum(abs(pc[i] - rc[i]) for i in range(n))
        pt_tot = sum(pc[:n]) or 1
        parity[r["format"]] = {
            "frame_match_pct": round(100 * same / n, 2),
            "avg_abs_diff_per_frame": round(diff / n, 3),
            "total_detection_diff_pct": round(
                100 * (r["total_detections"] - pt["total_detections"]) / pt_tot, 2
            ),
        }
    return parity


def print_table(results: list):
    print("\n\n" + "=" * 110)
    print("KARSILASTIRMA TABLOSU")
    print("=" * 110)
    hdr = f"{'Format':<10} {'Boyut':>8}  {'Avg ms':>8}  {'FPS':>8}  {'CPU%':>6}  {'GPU%':>6}  {'VRAM MB':>9}  {'Det':>6}  {'Conf':>7}"
    print(hdr)
    print("-" * 110)
    for r in results:
        gpu_u = f"{r['gpu_util_avg_pct']:.1f}" if r['gpu_util_avg_pct'] is not None else "N/A"
        gpu_m = f"{r['gpu_mem_avg_mb']:.0f}" if r['gpu_mem_avg_mb'] is not None else "N/A"
        print(f"{r['format'].upper():<10} "
              f"{r['model_size_mb']:>8.1f}  "
              f"{r['avg_inference_ms']:>8.2f}  "
              f"{r['fps']:>8.2f}  "
              f"{r['cpu_avg_pct']:>6.1f}  "
              f"{gpu_u:>6}  "
              f"{gpu_m:>9}  "
              f"{r['total_detections']:>6}  "
              f"{r['avg_confidence']:>7.4f}")

    if len(results) > 1:
        baseline = next((r for r in results if r["format"] == "pt"), results[0])
        base_fps = baseline["fps"] or 1
        print("\n--- HIZ FARKI (PT baseline) ---")
        for r in results:
            print(f"  {r['format'].upper():<10} : {r['fps']/base_fps:.2f}x")

        parity = compute_parity(results)
        if parity:
            print("\n--- DOGRULUK PARITESI (PT referans) ---")
            for fmt, p in parity.items():
                print(f"  {fmt.upper():<10} : "
                      f"frame-match {p['frame_match_pct']}% | "
                      f"ort. fark/frame {p['avg_abs_diff_per_frame']} | "
                      f"toplam detection farki {p['total_detection_diff_pct']}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--results", type=str, default="results/benchmark.json")
    args = parser.parse_args()

    candidates = {
        "pt":     Path(args.models_dir) / "best.pt",
        "onnx":   Path(args.models_dir) / "best.onnx",
        "engine": Path(args.models_dir) / "best.engine",
    }

    results = []
    for fmt, path in candidates.items():
        if path.exists():
            try:
                m = benchmark_model(str(path), args.video, args.save_output,
                                    "outputs", args.conf)
                results.append(m)
            except Exception as e:
                print(f"[HATA] {fmt}: {e}")
        else:
            print(f"[ATLA] {path} bulunamadi.")

    if results:
        print_table(results)
        os.makedirs(os.path.dirname(args.results), exist_ok=True)
        summary = [{k: v for k, v in r.items() if k != "per_frame_detections"}
                   for r in results]
        summary.append({"parity": compute_parity(results)})
        with open(args.results, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSonuclar kaydedildi: {args.results}")


if __name__ == "__main__":
    main()
