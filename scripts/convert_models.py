"""
convert_models.py
-----------------
YOLOv8 .pt modelini sırasıyla ONNX ve TensorRT (.engine) formatlarına dönüştürür.

Kullanım:
    python convert_models.py --weights models/best.pt --imgsz 640 --half

Çıktılar:
    models/best.onnx
    models/best.engine
"""

import argparse
import os
import sys
from pathlib import Path


def fix_cuda_env():
    """
    CUDA_VISIBLE_DEVICES boş string "" ise TensorRT export başarısız olur.
    torch import edilmeden ÖNCE environment'ı düzelt.
    """
    val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if val == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[FIX] CUDA_VISIBLE_DEVICES='' → '0' olarak düzeltildi.")


# Env fix torch import'tan ÖNCE yapılmalı
fix_cuda_env()

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def check_env():
    print("=" * 50)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print("=" * 50)


def convert(weights: str, imgsz: int = 640, half: bool = True, device: int = 0):
    weights = Path(weights)
    assert weights.exists(), f"Model bulunamadi: {weights}"

    check_env()

    print(f"\n[1/2] PT modeli yukleniyor: {weights}")
    model = YOLO(str(weights))

    # ── ONNX export ──────────────────────────────────────────────
    print("\n[ONNX] Export basliyor...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=True,
        dynamic=False,
    )
    print(f"[ONNX] Basarili: {onnx_path}")

    # ── TensorRT export ──────────────────────────────────────────
    # Gereksinim: NVIDIA GPU + TensorRT kurulu olmalı
    # NOT: torch cu130 kullanıyorsan CUDA_VISIBLE_DEVICES ortam değişkenini
    #      kontrol et. Boş bırakılmışsa script otomatik düzeltir.
    print("\n[TensorRT] Export basliyor (dakikalar surebilir)...")

    if not torch.cuda.is_available():
        print("[TensorRT] HATA: CUDA kullanilabilir degil.")
        print("           torch.cuda.is_available() = False")
        print("           Asagidaki adimlardan birini dene:")
        print("           1) export CUDA_VISIBLE_DEVICES=0")
        print("           2) Bu scripti yeniden calistir")
        return

    try:
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,        # FP16 — daha hizli, ufak dogruluk farki
            device=device,
            workspace=4,      # GB; GPU VRAM azsa 2 yap
        )
        print(f"[TensorRT] Basarili: {engine_path}")
    except Exception as e:
        print(f"[TensorRT] Export basarisiz: {e}")
        print("           -> TensorRT kurulumunu kontrol et: pip install tensorrt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", default=True,
                        help="FP16 kulllan (varsayilan: True)")
    parser.add_argument("--device", type=int, default=0, help="GPU id")
    args = parser.parse_args()

    convert(args.weights, args.imgsz, args.half, args.device)
