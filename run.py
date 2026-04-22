"""
run.py  —  YOLO Model Format Benchmark: PT vs ONNX vs TensorRT
==============================================================

Tek komutla her şeyi yapar:
    1. PT  → ONNX dönüşümü
    2. PT  → TensorRT engine dönüşümü  (GPU gerekli)
    3. Her formatı video üzerinde benchmark eder
    4. Karşılaştırma grafikleri üretir
    5. Sonuçları README.md'e yazar

Kullanım:
    python run.py

Gereksinimler (kurulum):
    pip install ultralytics onnx onnxsim opencv-python numpy matplotlib psutil nvidia-ml-py

    ONNX GPU için (cuDNN 9 + CUDA 12 gerekli):
        pip install onnxruntime-gpu==1.20.1

    TensorRT için:
        pip install tensorrt
"""

# ─── Sabitler — sadece burayı düzenle ────────────────────────────────────────
PT_MODEL    = "models/best.pt"       # Modelini buraya koy
VIDEO       = "input.mp4"           # Test videonu buraya koy
IMG_SIZE    = 640
HALF        = True                   # TensorRT FP16 (True önerilir)
CONF        = 0.25                   # Detection eşiği
WARMUP      = 5                      # Warm-up frame sayısı
OUTPUT_DIR  = "outputs"             # Annotasyonlu videolar
RESULTS_DIR = "results"             # JSON + grafikler
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, time, json, threading, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# CUDA_VISIBLE_DEVICES boş ise TensorRT çalışmaz — import öncesi düzelt
if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch'u ORT session açılmadan ÖNCE import et (CUDA DLL preload)
try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
    _GPU_NAME   = torch.cuda.get_device_name(0) if _TORCH_CUDA else "N/A"
except ImportError:
    torch = None
    _TORCH_CUDA = False
    _GPU_NAME   = "N/A"

import cv2
import numpy as np
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML = False
    _NVML_HANDLE = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — Ortam kontrolü
# ══════════════════════════════════════════════════════════════════════════════

def check_env():
    print("\n" + "─" * 60)
    print("  ORTAM BİLGİSİ")
    print("─" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    if torch:
        print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA        : {'✅  ' + _GPU_NAME if _TORCH_CUDA else '❌  Yok'}")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu_ok = "CUDAExecutionProvider" in providers
        print(f"  ONNX RT     : {ort.__version__}  |  GPU: {'✅' if gpu_ok else '❌ (CPU fallback)'}")
        if not gpu_ok:
            print("               ⚠  cuDNN 9 + CUDA 12 gerekli.")
            print("               →  pip install onnxruntime-gpu==1.20.1")
    except ImportError:
        print("  ONNX RT     : ❌ Kurulu değil  →  pip install onnxruntime-gpu")

    # TensorRT
    try:
        import tensorrt as trt
        print(f"  TensorRT    : {trt.__version__}  ✅")
    except ImportError:
        print("  TensorRT    : ❌ Kurulu değil  →  pip install tensorrt")

    print("─" * 60)
    assert Path(PT_MODEL).exists(), (
        f"\n❌  Model bulunamadı: {PT_MODEL}\n"
        f"   '{PT_MODEL}' dosyasını klasöre koy ve tekrar çalıştır."
    )
    assert Path(VIDEO).exists(), (
        f"\n❌  Video bulunamadı: {VIDEO}\n"
        f"   '{VIDEO}' dosyasını klasöre koy ve tekrar çalıştır."
    )
    print(f"  Model       : {PT_MODEL}  ({Path(PT_MODEL).stat().st_size/1e6:.1f} MB)")
    print(f"  Video       : {VIDEO}")
    print("─" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — Model dönüşümleri
# ══════════════════════════════════════════════════════════════════════════════

def convert_onnx() -> str | None:
    out = Path(PT_MODEL).with_suffix(".onnx")
    if out.exists():
        print(f"[ONNX]   Mevcut dosya kullanılıyor: {out}")
        return str(out)
    print("[ONNX]   Dönüşüm başlıyor...")
    model = YOLO(PT_MODEL)
    path = model.export(format="onnx", imgsz=IMG_SIZE, opset=12,
                        simplify=True, dynamic=False)
    print(f"[ONNX]   ✅  {path}")
    return path


def convert_engine() -> str | None:
    out = Path(PT_MODEL).with_suffix(".engine")
    if out.exists():
        print(f"[ENGINE] Mevcut dosya kullanılıyor: {out}")
        return str(out)
    if not _TORCH_CUDA:
        print("[ENGINE] ❌  CUDA yok — TensorRT atlanıyor.")
        print("             export CUDA_VISIBLE_DEVICES=0  ile tekrar dene.")
        return None
    print("[ENGINE] Dönüşüm başlıyor (dakikalar sürebilir)...")
    model = YOLO(PT_MODEL)
    try:
        path = model.export(format="engine", imgsz=IMG_SIZE,
                            half=HALF, device=0, workspace=4)
        print(f"[ENGINE] ✅  {path}")
        return path
    except Exception as e:
        print(f"[ENGINE] ❌  {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — Resource monitor (CPU / GPU / RAM)
# ══════════════════════════════════════════════════════════════════════════════

class ResourceMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.running  = False
        self.cpu, self.ram, self.gpu_u, self.gpu_m = [], [], [], []
        self._proc   = psutil.Process(os.getpid())
        self._thread = None

    def _loop(self):
        self._proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        while self.running:
            time.sleep(self.interval)
            try:
                self.cpu.append(psutil.cpu_percent(interval=None))
                self.ram.append(self._proc.memory_info().rss / 1e6)
                if _NVML and _NVML_HANDLE:
                    u = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                    m = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                    self.gpu_u.append(u.gpu)
                    self.gpu_m.append(m.used / 1e6)
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

        def _a(lst): return round(float(np.mean(lst)), 2) if lst else None
        def _x(lst): return round(float(np.max(lst)),  2) if lst else None

        return {
            "cpu_avg_pct":      _a(self.cpu)   or 0.0,
            "cpu_max_pct":      _x(self.cpu)   or 0.0,
            "ram_avg_mb":       _a(self.ram)   or 0.0,
            "ram_max_mb":       _x(self.ram)   or 0.0,
            "gpu_util_avg_pct": _a(self.gpu_u),
            "gpu_util_max_pct": _x(self.gpu_u),
            "gpu_mem_avg_mb":   _a(self.gpu_m),
            "gpu_mem_max_mb":   _x(self.gpu_m),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — Tek model benchmark
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_one(model_path: str) -> dict | None:
    fmt = Path(model_path).suffix.lstrip(".")
    label = fmt.upper()
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {model_path}  [{label}]")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"  ❌  Model yüklenemedi: {e}")
        return None

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"  ❌  Video açılamadı: {VIDEO}")
        return None

    fps_v  = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size   = Path(model_path).stat().st_size / 1e6

    # Çıktı video
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_p  = os.path.join(OUTPUT_DIR, f"output_{fmt}.mp4")
    writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (W, H))

    # Warm-up
    print(f"  Isınma ({WARMUP} frame)...")
    ret, frame = cap.read()
    if ret:
        for _ in range(WARMUP):
            try:
                _ = model(frame, verbose=False, conf=CONF)
            except Exception:
                pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    inf, confs, per_frame = [], [], []
    n_det_total = 0
    monitor = ResourceMonitor()
    monitor.start()
    t_start = time.time()
    fc = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        try:
            results = model(frame, verbose=False, conf=CONF)
        except Exception as e:
            print(f"  ❌  Inference hatası (frame {fc}): {e}")
            cap.release(); writer.release()
            monitor.stop()
            return None

        inf.append((time.time() - t0) * 1000)
        r  = results[0]
        nd = len(r.boxes) if r.boxes is not None else 0
        n_det_total += nd
        per_frame.append(nd)
        if nd > 0 and r.boxes.conf is not None:
            confs.extend(r.boxes.conf.cpu().numpy().tolist())
        writer.write(r.plot())
        fc += 1
        if fc % 30 == 0:
            print(f"  Frame {fc:>4}/{N}  |  {inf[-1]:.1f} ms  |  det: {nd}")

    total_t = time.time() - t_start
    res     = monitor.stop()
    cap.release(); writer.release()

    avg = float(np.mean(inf))
    metrics = {
        "format":                  fmt,
        "model_path":              model_path,
        "model_size_mb":           round(size, 2),
        "total_frames":            fc,
        "total_time_sec":          round(total_t, 2),
        "avg_inference_ms":        round(avg, 2),
        "median_inference_ms":     round(float(np.median(inf)), 2),
        "p95_inference_ms":        round(float(np.percentile(inf, 95)), 2),
        "min_inference_ms":        round(float(np.min(inf)), 2),
        "max_inference_ms":        round(float(np.max(inf)), 2),
        "fps":                     round(1000.0 / avg, 2) if avg > 0 else 0,
        "total_detections":        n_det_total,
        "avg_detections_per_frame":round(n_det_total / max(fc, 1), 2),
        "avg_confidence":          round(float(np.mean(confs)), 4) if confs else 0.0,
        "_per_frame":              per_frame,
        **res,
    }

    print(f"\n  {'─'*40}")
    skip = {"_per_frame", "model_path", "format"}
    for k, v in metrics.items():
        if k not in skip:
            bar = ""
            if k == "fps":
                bar = "  ★" if v > 100 else ""
            print(f"  {k:<28}: {v}{bar}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — Karşılaştırma tablosu
# ══════════════════════════════════════════════════════════════════════════════

def parity(results: list) -> dict:
    pt = next((r for r in results if r["format"] == "pt"), None)
    if not pt:
        return {}
    out = {}
    for r in results:
        if r["format"] == "pt":
            continue
        pc, rc = pt["_per_frame"], r["_per_frame"]
        n = min(len(pc), len(rc))
        if n == 0:
            continue
        same = sum(1 for i in range(n) if pc[i] == rc[i])
        diff = sum(abs(pc[i] - rc[i]) for i in range(n))
        pt_t = sum(pc[:n]) or 1
        out[r["format"]] = {
            "frame_match_pct":        round(100 * same / n, 2),
            "avg_abs_diff_per_frame": round(diff / n, 3),
            "total_det_diff_pct":     round(100 * (r["total_detections"] - pt["total_detections"]) / pt_t, 2),
        }
    return out


def print_table(results: list, par: dict):
    print("\n\n" + "═" * 105)
    print("  KARŞILAŞTIRMA TABLOSU")
    print("═" * 105)
    h = (f"  {'Format':<10} {'Boyut':>8}  {'Avg ms':>8}  {'p95 ms':>8}  "
         f"{'FPS':>8}  {'CPU%':>6}  {'GPU%':>6}  {'VRAM MB':>9}  {'Det':>6}  {'Conf':>7}")
    print(h)
    print("─" * 105)
    for r in results:
        gu = f"{r['gpu_util_avg_pct']:.1f}" if r["gpu_util_avg_pct"] is not None else "N/A"
        gm = f"{r['gpu_mem_avg_mb']:.0f}"   if r["gpu_mem_avg_mb"]  is not None else "N/A"
        print(f"  {r['format'].upper():<10}"
              f" {r['model_size_mb']:>8.1f}"
              f"  {r['avg_inference_ms']:>8.2f}"
              f"  {r['p95_inference_ms']:>8.2f}"
              f"  {r['fps']:>8.2f}"
              f"  {r['cpu_avg_pct']:>6.1f}"
              f"  {gu:>6}"
              f"  {gm:>9}"
              f"  {r['total_detections']:>6}"
              f"  {r['avg_confidence']:>7.4f}")

    # Hız farkı
    baseline = next((r for r in results if r["format"] == "pt"), results[0])
    base_fps = baseline["fps"] or 1
    print("\n  HIZ FARKI (PT = 1.00×)")
    for r in results:
        arrow = "↑" if r["fps"] / base_fps > 1 else ("↓" if r["fps"] / base_fps < 1 else "=")
        print(f"    {r['format'].upper():<10} : {r['fps']/base_fps:.2f}×  {arrow}")

    # Accuracy parity
    if par:
        print("\n  DOĞRULUK PARİTESİ  (PT referans — toplam det. farkı)")
        for fmt, p in par.items():
            print(f"    {fmt.upper():<10} : "
                  f"frame-match {p['frame_match_pct']:>6.2f}%  |  "
                  f"ort. fark/frame {p['avg_abs_diff_per_frame']:.3f}  |  "
                  f"toplam fark {p['total_det_diff_pct']:+.2f}%")
    print("═" * 105)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — Grafikler
# ══════════════════════════════════════════════════════════════════════════════

_CLR = {"PT": "#EE4C2C", "ONNX": "#005CED", "ENGINE": "#76B900"}


def _bar(ax, labels, values, title, ylabel, fmt="{:.1f}", ylim_pct=False):
    cols = [_CLR.get(l, "#888") for l in labels]
    valid = [(l, v, c) for l, v, c in zip(labels, values, cols) if v is not None]
    if not valid:
        ax.set_visible(False); return
    ls, vs, cs = zip(*valid)
    bars = ax.bar(ls, vs, color=cs, width=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    if ylim_pct:
        ax.set_ylim(0, 110)
    mx = max(vs)
    for b, v in zip(bars, vs):
        ax.text(b.get_x() + b.get_width() / 2,
                v + mx * 0.02, fmt.format(v),
                ha="center", va="bottom", fontsize=9, fontweight="bold")


def make_plots(results: list, par: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels = [r["format"].upper() for r in results]
    bg = "#f8f9fa"

    # ── Performans ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(bg)
    for ax in axes: ax.set_facecolor(bg)
    _bar(axes[0], labels, [r["fps"] for r in results],             "FPS ↑",               "frame/s")
    _bar(axes[1], labels, [r["avg_inference_ms"] for r in results],"Ort. Inference ↓",    "ms")
    _bar(axes[2], labels, [r["p95_inference_ms"] for r in results],"p95 Inference ↓",     "ms")
    _bar(axes[3], labels, [r["model_size_mb"] for r in results],   "Model Boyutu",        "MB")
    fig.suptitle("Performans: PT vs ONNX vs TensorRT", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_performance.png", dpi=150,
                bbox_inches="tight", facecolor=bg)
    plt.close()

    # ── Kaynak kullanımı ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(bg)
    for ax in axes: ax.set_facecolor(bg)
    _bar(axes[0], labels, [r["cpu_avg_pct"] for r in results],           "CPU Kullanımı ↓",  "%")
    _bar(axes[1], labels, [r["ram_avg_mb"] for r in results],            "RAM Kullanımı ↓",  "MB")
    _bar(axes[2], labels, [r.get("gpu_util_avg_pct") for r in results],  "GPU Utilization",  "%")
    _bar(axes[3], labels, [r.get("gpu_mem_avg_mb") for r in results],    "GPU VRAM",         "MB")
    fig.suptitle("Kaynak Kullanımı: PT vs ONNX vs TensorRT", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_resources.png", dpi=150,
                bbox_inches="tight", facecolor=bg)
    plt.close()

    # ── Doğruluk ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(bg)
    for ax in axes: ax.set_facecolor(bg)
    _bar(axes[0], labels, [r["total_detections"] for r in results],
         "Toplam Detection", "adet", fmt="{:.0f}")
    _bar(axes[1], labels, [r["avg_confidence"] for r in results],
         "Ort. Confidence",  "0–1",  fmt="{:.4f}")
    if par:
        pl = [k.upper() for k in par]
        pm = [par[k]["frame_match_pct"] for k in par]
        _bar(axes[2], pl, pm, "Frame-Match (PT ref.)", "%",
             fmt="{:.1f}%", ylim_pct=True)
        axes[2].text(0.5, -0.16,
            "Her frame'de PT ile aynı detection sayısı yüzdesi.",
            ha="center", va="top", transform=axes[2].transAxes,
            fontsize=8, color="#555", style="italic")
    else:
        axes[2].set_visible(False)
    fig.suptitle("Doğruluk (Accuracy Parity): PT vs ONNX vs TensorRT",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_accuracy.png", dpi=150,
                bbox_inches="tight", facecolor=bg)
    plt.close()

    print(f"\n  Grafikler kaydedildi → {RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — README üret
# ══════════════════════════════════════════════════════════════════════════════

def write_readme(results: list, par: dict):
    pt  = next((r for r in results if r["format"] == "pt"),     None)
    onnx= next((r for r in results if r["format"] == "onnx"),   None)
    eng = next((r for r in results if r["format"] == "engine"), None)

    def row(r):
        if r is None:
            return "| — | — | — | — | — | — | — |"
        gu = f"{r['gpu_util_avg_pct']:.1f}%" if r["gpu_util_avg_pct"] is not None else "N/A"
        gm = f"{r['gpu_mem_avg_mb']:.0f}"    if r["gpu_mem_avg_mb"]  is not None else "N/A"
        base_fps = (pt["fps"] if pt else 1) or 1
        spd = f"{r['fps']/base_fps:.2f}×"
        return (f"| **{r['format'].upper()}** "
                f"| {r['model_size_mb']} MB "
                f"| {r['avg_inference_ms']} ms "
                f"| {r['fps']} "
                f"| {r['cpu_avg_pct']}% / {gu} "
                f"| {gm} MB "
                f"| {spd} |")

    def acc_row(r, p):
        if r is None:
            return "| — | — | — | — |"
        if r["format"] == "pt":
            return (f"| **PT** | {r['total_detections']} | "
                    f"{r['avg_confidence']:.4f} | referans |")
        pm = f"{p.get('frame_match_pct', 'N/A')}%" if p else "N/A"
        dd = f"{p.get('total_det_diff_pct', 'N/A'):+.2f}%" if p else "N/A"
        return (f"| **{r['format'].upper()}** | {r['total_detections']} | "
                f"{r['avg_confidence']:.4f} | {pm} eşleşme / {dd} fark |")

    onnx_warn = ""
    if onnx is None:
        onnx_warn = (
            "\n> ⚠️  **ONNX bu çalıştırmada başarısız oldu.**  \n"
            "> `libcudnn_adv.so.9` bulunamadı — cuDNN 9 kurulu değil.  \n"
            "> Çözüm: `pip install onnxruntime-gpu==1.20.1` + cuDNN 9 kur.  \n"
        )

    md = f"""# YOLO Model Formatları Karşılaştırması: PT vs ONNX vs TensorRT

> `.pt`, `.onnx` ve `.engine` formatlarını hız · boyut · GPU/CPU kullanımı · doğruluk açısından karşılaştıran benchmark projesi.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📖 İçindekiler

1. [Formatların Tanımları](#-formatların-tanımları)
2. [Dönüşüm Akışı](#-dönüşüm-akışı)
3. [Kurulum](#-kurulum)
4. [Kullanım](#-kullanım)
5. [Benchmark Sonuçları](#-benchmark-sonuçları)
6. [Sonuçların Yorumu](#-sonuçların-yorumu)
7. [Ne Zaman Hangi Format?](#-ne-zaman-hangi-format)
8. [Bilinen Sorunlar](#-bilinen-sorunlar)

---

## 🔍 Formatların Tanımları

### 1️⃣ PyTorch — `.pt`

Meta (Facebook) tarafından geliştirilen açık kaynaklı derin öğrenme framework'ünün model kayıt formatı.  
`.pt` dosyası model ağırlıklarını ve mimarisini içerir.

| | |
|---|---|
| **Güçlü yönleri** | Eğitim için birincil ortam · Debug kolaylığı · Tam Python entegrasyonu |
| **Zayıf yönleri** | Production'da en yavaş · Ağır runtime · Mobil/edge için uygun değil |

---

### 2️⃣ ONNX — `.onnx`

ONNX (Open Neural Network Exchange), ML modelleri için **framework-bağımsız** açık ara format standardı.  
"Modeller için evrensel dil" — PyTorch'ta eğit, istediğin runtime'da çalıştır.

| | |
|---|---|
| **Güçlü yönleri** | Cross-platform · CPU/GPU/mobil/tarayıcı uyumu · Geniş ekosistem |
| **Zayıf yönleri** | CUDA sürüm uyumsuzluğunda sessizce CPU'ya gerileme · TensorRT kadar hızlı değil |

> ⚠️ `onnxruntime-gpu` **cuDNN 9 + CUDA 12** gerektirir. CUDA 13 (cu130) ile GPU çalışmaz.

---

### 3️⃣ TensorRT — `.engine`

NVIDIA'nın inference optimizer ve runtime'ı. Modeli **hedef GPU için** özel olarak derler.

| | |
|---|---|
| **Güçlü yönleri** | En yüksek FPS · Layer fusion · FP16/INT8 · Kernel auto-tuning |
| **Zayıf yönleri** | Yalnızca NVIDIA GPU · `.engine` dosyası taşınamaz · Build süresi uzun |

---

## 🔄 Dönüşüm Akışı

```
best.pt  ──export──▶  best.onnx  ──export──▶  best.engine
(Eğitim)             (Taşınabilir)            (GPU'da en hızlı)
```

---

## ⚙️ Kurulum

```bash
git clone https://github.com/KULLANICI_ADIN/model-comparison.git
cd model-comparison
pip install ultralytics onnx onnxsim opencv-python numpy matplotlib psutil nvidia-ml-py

# ONNX GPU (cuDNN 9 + CUDA 12 gerekli):
pip install onnxruntime-gpu==1.20.1

# TensorRT:
pip install tensorrt
```

Sonra `models/best.pt` ve `input.mp4` dosyalarını koy.

---

## 🚀 Kullanım

```bash
python run.py
```

Tek komut her şeyi yapar:
dönüşüm → benchmark → grafikler → README güncelleme.

Sabitler `run.py` dosyasının başında düzenlenebilir:

```python
PT_MODEL = "models/best.pt"
VIDEO    = "input.mp4"
IMG_SIZE = 640
HALF     = True
CONF     = 0.25
```

---

## 📊 Benchmark Sonuçları
{onnx_warn}
### Performans

| Format | Boyut | Avg ms | FPS | CPU / GPU | VRAM | Hız |
|--------|------:|------:|----:|----------:|-----:|----:|
{row(pt)}
{row(onnx)}
{row(eng)}

### Doğruluk

| Format | Toplam Det. | Ort. Conf | Parity (PT'ye göre) |
|--------|----------:|----------:|---------------------|
{acc_row(pt,  {})}
{acc_row(onnx, par.get("onnx", {}))}
{acc_row(eng,  par.get("engine", {}))}

### Grafikler

| Performans | Kaynaklar | Doğruluk |
|:---:|:---:|:---:|
| ![perf](results/comparison_performance.png) | ![res](results/comparison_resources.png) | ![acc](results/comparison_accuracy.png) |

---

## 🔍 Sonuçların Yorumu

### Hız

TensorRT, PT'ye göre **~{f'{eng["fps"]/pt["fps"]:.2f}×' if pt and eng else 'N/A'}** daha hızlı.  
GPU tüm iş yükünü alarak CPU'yu serbest bırakıyor (%{eng['cpu_avg_pct'] if eng else 'N/A'} kullanım) — aynı makinede başka servislerin çalışmasına imkân tanıyor.

### Doğruluk

TensorRT FP16 modunda çalışmasına rağmen PT ile neredeyse özdeş sonuç üretiyor  
(toplam detection farkı yalnızca {par.get("engine", {{}}).get("total_det_diff_pct", "N/A")}%). Kritik uygulamalarda FP32 tercih edilebilir.

### ONNX

ONNX'in asıl avantajı **taşınabilirlik** ve **cross-platform** desteği.  
GPU doğru kurulduğunda PT'ye göre belirgin şekilde hızlı, TensorRT'ye yakın performans verir.

---

## 🎯 Ne Zaman Hangi Format?

| Senaryo | Format |
|---------|--------|
| Model eğitimi / araştırma | **PT** |
| Cross-platform deployment | **ONNX** |
| Mobil / edge cihazlar | **ONNX** |
| NVIDIA GPU production | **TensorRT** |
| Gerçek zamanlı video | **TensorRT** |
| CPU-only sunucu | **ONNX** |

---

## 🐛 Bilinen Sorunlar

### ONNX — `libcudnn_adv.so.9: cannot open shared object file`

`onnxruntime-gpu` **cuDNN 9 + CUDA 12** gerektirir. CUDA 13 ile resmi destek yok.

**Çözüm A — PyTorch'u CUDA 12'ye geç (önerilen):**
```bash
pip uninstall torch torchvision onnxruntime-gpu -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

**Çözüm B — ORT CUDA 13 nightly (deneysel):**
```bash
pip install flatbuffers numpy packaging protobuf sympy
pip install --pre --index-url \\
  https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \\
  onnxruntime-gpu
```

### TensorRT — `CUDA_VISIBLE_DEVICES` boş

```bash
export CUDA_VISIBLE_DEVICES=0
python run.py
```

### `.engine` başka makinede çalışmıyor

Normal — GPU mimarisine özeldir. Her makinede `run.py` yeniden çalıştır.

---

## 📁 Proje Yapısı

```
model-comparison/
├── run.py                        ← Tek giriş noktası
├── models/
│   ├── best.pt
│   ├── best.onnx                 ← otomatik oluşturulur
│   └── best.engine               ← otomatik oluşturulur
├── outputs/
│   ├── output_pt.mp4
│   ├── output_onnx.mp4
│   └── output_engine.mp4
├── results/
│   ├── benchmark.json
│   ├── comparison_performance.png
│   ├── comparison_resources.png
│   └── comparison_accuracy.png
└── README.md                     ← otomatik güncellenir
```

---

**Yazan:** [İsmin]  
**İletişim:** [email / linkedin]
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("  README.md güncellendi ✅")


# ══════════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 60)
    print("  YOLO FORMAT BENCHMARK  —  PT / ONNX / TensorRT")
    print("█" * 60)

    # 0. Ortam kontrolü
    check_env()

    # 1. Dönüşümler
    print("[ ADIM 1/4 ]  Model dönüşümleri")
    onnx_path   = convert_onnx()
    engine_path = convert_engine()

    # 2. Benchmark
    print("\n[ ADIM 2/4 ]  Benchmark")
    candidates = {
        "pt":     Path(PT_MODEL).with_suffix(".pt"),
        "onnx":   Path(PT_MODEL).with_suffix(".onnx"),
        "engine": Path(PT_MODEL).with_suffix(".engine"),
    }
    results = []
    for fmt, path in candidates.items():
        if path.exists():
            m = benchmark_one(str(path))
            if m:
                results.append(m)
        else:
            print(f"\n  [ATLA] {path} bulunamadı.")

    if not results:
        print("❌  Hiçbir model çalıştırılamadı.")
        sys.exit(1)

    # 3. Tablo + parity
    par = parity(results)
    print_table(results, par)

    # 4. JSON kaydet
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = [{k: v for k, v in r.items() if k != "_per_frame"} for r in results]
    summary.append({"parity": par})
    json_path = os.path.join(RESULTS_DIR, "benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[ ADIM 3/4 ]  JSON → {json_path}")

    # 5. Grafikler
    print("\n[ ADIM 4/4 ]  Grafikler")
    make_plots(results, par)

    # 6. README
    write_readme(results, par)

    print("\n" + "█" * 60)
    print("  TAMAMLANDI ✅")
    print(f"  Grafikler  : {RESULTS_DIR}/comparison_*.png")
    print(f"  JSON       : {json_path}")
    print(f"  Videolar   : {OUTPUT_DIR}/output_*.mp4")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
