"""
plot_results.py
---------------
benchmark.json sonuçlarından 3 farklı grafik üretir.

Üretilen dosyalar:
    results/comparison_performance.png  — FPS, inference süresi, model boyutu
    results/comparison_resources.png    — CPU, GPU utilization, RAM, VRAM
    results/comparison_accuracy.png     — Detection sayısı, confidence, parity

Kullanım:
    python plot_results.py --input results/benchmark.json
"""

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Format renkleri
COLORS = {"PT": "#EE4C2C", "ONNX": "#005CED", "ENGINE": "#76B900"}


def _bar(ax, labels, values, title, ylabel, fmt="{:.1f}", ylim_pct=False):
    cols = [COLORS.get(l, "#888888") for l in labels]
    valid = [(l, v, c) for l, v, c in zip(labels, values, cols) if v is not None]
    if not valid:
        ax.axis("off")
        return
    ls, vs, cs = zip(*valid)
    bars = ax.bar(ls, vs, color=cs, width=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    if ylim_pct:
        ax.set_ylim(0, 110)
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vs) * 0.02,
                fmt.format(v), ha="center", va="bottom", fontsize=9, fontweight="bold")


def plot_performance(data, out_dir):
    labels = [d["format"].upper() for d in data]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")

    _bar(axes[0], labels, [d["fps"] for d in data], "FPS ↑", "frame/s")
    _bar(axes[1], labels, [d["avg_inference_ms"] for d in data],
         "Ort. Inference ↓", "ms")
    _bar(axes[2], labels, [d["p95_inference_ms"] for d in data],
         "p95 Inference ↓", "ms")
    _bar(axes[3], labels, [d["model_size_mb"] for d in data],
         "Model Boyutu", "MB")

    fig.suptitle("Performans: PT vs ONNX vs TensorRT", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "comparison_performance.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print(f"  -> {p}")


def plot_resources(data, out_dir):
    labels = [d["format"].upper() for d in data]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")

    _bar(axes[0], labels, [d["cpu_avg_pct"] for d in data],
         "CPU Kullanımı (ort.) ↓", "%")
    _bar(axes[1], labels, [d["ram_avg_mb"] for d in data],
         "RAM Kullanımı (ort.) ↓", "MB")
    _bar(axes[2], labels,
         [d.get("gpu_util_avg_pct") for d in data],
         "GPU Utilization (ort.)", "%")
    _bar(axes[3], labels,
         [d.get("gpu_mem_avg_mb") for d in data],
         "GPU VRAM (ort.)", "MB")

    fig.suptitle("Kaynak Kullanımı: PT vs ONNX vs TensorRT", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "comparison_resources.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print(f"  -> {p}")


def plot_accuracy(data, parity, out_dir):
    labels = [d["format"].upper() for d in data]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")

    _bar(axes[0], labels, [d["total_detections"] for d in data],
         "Toplam Detection Sayısı", "adet", fmt="{:.0f}")
    _bar(axes[1], labels, [d["avg_confidence"] for d in data],
         "Ortalama Confidence", "0–1", fmt="{:.4f}")

    if parity:
        pl = [k.upper() for k in parity]
        pm = [parity[k]["frame_match_pct"] for k in parity]
        _bar(axes[2], pl, pm,
             "Frame-Eşleşme (PT referans)", "%", fmt="{:.1f}%", ylim_pct=True)
        # Açıklama notu
        axes[2].text(0.5, -0.18,
                     "Frame-match: Her frame'de PT ile aynı sayıda\n"
                     "tespit yapılma yüzdesi. 100% = mükemmel parity.",
                     ha="center", va="top", transform=axes[2].transAxes,
                     fontsize=8, color="#555555", style="italic")
    else:
        axes[2].axis("off")

    fig.suptitle("Doğruluk (Accuracy Parity): PT vs ONNX vs TensorRT",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "comparison_accuracy.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print(f"  -> {p}")


def main(results_path, out_dir):
    with open(results_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data, parity = [], {}
    for item in raw:
        if "parity" in item and len(item) == 1:
            parity = item["parity"]
        else:
            data.append(item)

    if not data:
        print("Sonuc dosyasi bos!")
        return

    os.makedirs(out_dir, exist_ok=True)
    print("Grafikler olusturuluyor:")
    plot_performance(data, out_dir)
    plot_resources(data, out_dir)
    plot_accuracy(data, parity, out_dir)
    print("Tamamlandi!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/benchmark.json")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    main(args.input, args.out)
