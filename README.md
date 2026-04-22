# YOLO Model Formatları Karşılaştırması: PT vs ONNX vs TensorRT

> Aynı YOLO modelini `.pt`, `.onnx` ve `.engine` formatlarında çalıştırıp **hız, boyut, GPU/CPU kullanımı ve doğruluk** açısından karşılaştıran bir benchmark projesi.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📖 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Formatların Tanımları](#-formatların-tanımları)
- [Ölçülen Metrikler](#-ölçülen-metrikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Gerçek Benchmark Sonuçları](#-gerçek-benchmark-sonuçları)
- [Sonuçların Yorumu](#-sonuçların-yorumu)
- [Hangi Formatı Ne Zaman Kullanmalıyım?](#-hangi-formatı-ne-zaman-kullanmalıyım)
- [Bilinen Sorunlar & Çözümler](#-bilinen-sorunlar--çözümler)

---

## 🎯 Proje Hakkında

Bu projede bir **YOLO object detection** modelini (`best.pt`) 3 formata dönüştürüp aynı video üzerinde karşılaştırıyoruz:

| Format | Uzantı | Kullanım Alanı |
|--------|--------|----------------|
| PyTorch | `.pt` / `.pth` | Eğitim ve geliştirme |
| ONNX | `.onnx` | Cross-platform inference |
| TensorRT | `.engine` / `.trt` | NVIDIA GPU'da maksimum hız |

---

## 🔍 Formatların Tanımları

### 1️⃣ PyTorch (`.pt`)

**Nedir?**
Meta (Facebook) tarafından geliştirilen açık kaynaklı derin öğrenme framework'ünün model kayıt formatıdır. `.pt` dosyaları ağırlıkları ve/veya model mimarisini içerir.

**Avantajları:**
- Model eğitiminde birincil ortam
- Dinamik hesaplama grafiği → debug çok kolay
- Araştırma ve prototipleme için ideal
- Python ekosistemiyle tam entegrasyon

**Dezavantajları:**
- TensorRT'ye kıyasla daha yavaş inference
- Python + PyTorch runtime bağımlılığı (ağır)
- Mobil/edge deployment için uygun değil

---

### 2️⃣ ONNX (`.onnx`)

**Nedir?**
ONNX (Open Neural Network Exchange), ML modelleri için **framework-bağımsız** açık bir ara format standardıdır. PyTorch → ONNX → herhangi bir runtime zinciriyle çalışır.

**Avantajları:**
- **Taşınabilirlik**: PyTorch'ta eğit, TensorFlow/OpenVINO/TVM'de çalıştır
- **Cross-platform**: Windows, Linux, macOS, mobil, tarayıcı (ONNX.js)
- **Donanım esnekliği**: CPU, NVIDIA GPU, AMD GPU, Intel, ARM
- GPU sürücüsü doğru kuruluysa PyTorch'a yakın veya daha hızlı olabilir

**Dezavantajları:**
- Bazı özel PyTorch operatörleri desteklenmeyebilir
- TensorRT kadar agresif GPU optimizasyonu yapamaz
- PT'ye göre daha büyük dosya boyutu (~2×)
- CUDA sürümü uyumsuzluğunda CPU'ya sessizce gerileme (fallback) yapar → hız kaybı

> ℹ️ **CUDA Uyumluluğu:** `onnxruntime-gpu` resmi olarak **CUDA 12.x**'i destekler. Bu benchmarkta `torch-2.5.1+cu121` + `onnxruntime-gpu 1.23.2` kombinasyonuyla `CUDAExecutionProvider` başarıyla kullanıldı.

---

### 3️⃣ TensorRT Engine (`.engine`)

**Nedir?**
NVIDIA'nın geliştirdiği yüksek performanslı inference optimizer ve runtime'ıdır. Modeli **hedef GPU mimarisi için** özel olarak derler ve optimizasyon yapar.

**Avantajları:**
- **3 format arasında en yüksek FPS** (NVIDIA GPU'larda)
- **Layer fusion**: Birden fazla operasyonu tek CUDA kernel'e birleştirir
- **Precision calibration**: FP32 → FP16 → INT8 ile dramatik hızlanma
- **Kernel auto-tuning**: GPU'ya özel en verimli algoritmayı otomatik seçer
- Gerçek zamanlı video analitik, otonom sistemler, edge AI için ideal

**Dezavantajları:**
- ⚠️ **Sadece NVIDIA GPU'larda çalışır**
- `.engine` dosyası GPU modeline ve TensorRT versiyonuna özeldir — farklı makinede çalışmaz, yeniden build gerekir
- İlk build süresi uzun (bu benchmarkta ~142 saniye)
- FP16 modunda ufak doğruluk farkı olabilir (genellikle ihmal edilebilir)

---

## 🔄 Dönüşüm Akışı

```
  ┌──────────┐    export     ┌──────────┐    export    ┌──────────────┐
  │ best.pt  │ ────────────▶ │best.onnx │ ───────────▶ │ best.engine  │
  └──────────┘               └──────────┘              └──────────────┘
   (Eğitim)                 (Taşınabilir)              (GPU'da en hızlı)
```

---

## 📊 Ölçülen Metrikler

| Kategori | Metrik | Açıklama |
|----------|--------|----------|
| **Hız** | `avg_inference_ms` | Frame başına ortalama inference süresi |
| | `p95_inference_ms` | %95'lik dilim — gerçek zamanlı worst-case |
| | `fps` | Saniyede işlenen frame |
| **Boyut** | `model_size_mb` | Disk dosyası boyutu (MB) |
| **Kaynak** | `cpu_avg_pct` | CPU kullanım yüzdesi (ort.) |
| | `ram_avg_mb` | İşlem RAM kullanımı (ort.) |
| | `gpu_util_avg_pct` | GPU utilization % (NVIDIA) |
| | `gpu_mem_avg_mb` | GPU VRAM kullanımı (ort.) |
| **Doğruluk** | `total_detections` | Toplam nesne tespiti |
| | `avg_confidence` | Tespitlerin ortalama confidence skoru |
| | `frame_match_pct` | PT ile aynı sayıda tespit yapılan frame oranı |

---

## ⚙️ Kurulum

### 1. Repo'yu klonla
```bash
git clone https://github.com/KULLANICI_ADI/model-comparison.git
cd model-comparison
```

### 2. Sanal ortam oluştur
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
# veya: venv\Scripts\activate  (Windows)
```

### 3. Bağımlılıkları kur

**CUDA 12.x kullananlar (standart — önerilen):**
```bash
pip install -r requirements.txt
pip install onnxruntime-gpu
```

### 4. TensorRT kur (GPU gerekli)
```bash
pip install tensorrt
```

### 5. Dosyaları yerleştir
```
model-comparison/
├── models/
│   └── best.pt        ← Kendi modelini koy
└── input.mp4          ← Test videonu koy
```

---

## 🚀 Kullanım

### Adım 1: Dönüştür
```bash
python scripts/convert_models.py --weights models/best.pt --imgsz 640 --half
```
→ `models/best.onnx` ve `models/best.engine` oluşturulur.

### Adım 2: Benchmark
```bash
python scripts/benchmark.py --video input.mp4 --save_output
```
→ Her format çalıştırılır, terminale tablo yazdırılır, `results/benchmark.json` kaydedilir.

### Adım 3: Grafik
```bash
python scripts/plot_results.py --input results/benchmark.json
```
→ 3 grafik oluşturulur:
- `results/comparison_performance.png`
- `results/comparison_resources.png`
- `results/comparison_accuracy.png`

---

## 📊 Gerçek Benchmark Sonuçları

Aşağıdaki sonuçlar bu repo'daki model ve test videosu kullanılarak elde edilmiştir.

**Test ortamı:**
```
Python      : 3.10.20
PyTorch     : 2.5.1+cu121
CUDA        : NVIDIA GeForce RTX 4060 Laptop GPU
ONNX RT     : 1.23.2  |  GPU: ✅ CUDAExecutionProvider
TensorRT    : 10.16.1.11
Model       : models/best.pt  (6.3 MB)
Video       : input.mp4  —  3954 frame
```

### Performans

| Format | Boyut (MB) | Avg (ms) | Median (ms) | p95 (ms) | FPS | Hızlanma |
|--------|----------:|--------:|----------:|--------:|----:|---------:|
| **PT** | 6.3 | 5.10 | 4.89 | 6.97 | 196.14 | 1.00× |
| **ONNX** | 12.3 | 5.63 | 5.47 | 7.13 | 177.55 | 0.91× |
| **ENGINE** | 7.7 | **3.62** | **3.46** | **5.20** | **276.00** | **1.41×** |

> ONNX bu testte `CUDAExecutionProvider` ile çalıştı. PT'den hafif yavaş çıkmasının sebebi ONNX Runtime'ın opset 12 ile derlenmesi ve küçük GPU scheduling overhead'i.

### Kaynak Kullanımı

| Format | CPU avg | CPU max | GPU avg | GPU max | RAM (MB) | VRAM (MB) |
|--------|--------:|--------:|--------:|--------:|---------:|----------:|
| **PT** | 13.3% | 47.9% | 22.2% | 40.0% | 3342 | 790 |
| **ONNX** | 14.0% | 73.8% | 30.1% | 33.0% | 3460 | 926 |
| **ENGINE** | 14.6% | 33.8% | 17.7% | 31.0% | 3520 | 838 |

### Doğruluk (Accuracy Parity)

| Format | Toplam Det. | Ort. Conf | Frame-match | Det. Farkı |
|--------|----------:|----------:|------------:|----------:|
| **PT** | 22.199 | 0.7893 | referans | — |
| **ONNX** | 21.865 | 0.7940 | %85.26 | −1.50% |
| **ENGINE** | 21.870 | 0.7939 | %85.53 | −1.48% |

> Frame-match %85'in üzerinde; toplam detection farkı ise yalnızca ~%1.5. ONNX ve ENGINE neredeyse aynı tespitleri yapıyor ve confidence skorları PT ile örtüşüyor. TensorRT FP16 modunda çalışmasına karşın doğruluk kaybı ihmal edilebilir düzeyde.

---

## 🔍 Sonuçların Yorumu

### Hız

**TensorRT** açık ara en hızlı: PT'ye göre **1.41×**, ONNX'e göre **1.55×** daha hızlı, 276 FPS ile gerçek zamanlı her senaryoyu rahatlıkla karşılıyor.

**ONNX**, GPU'da çalışmasına rağmen PT'den %9 geride kaldı (0.91×). Bunun temel nedeni opset farkı ve ONNX Runtime'ın PyTorch kadar optimize CUDA kernel'leri kullanmaması. Yine de 177 FPS ile production için yeterli bir hız.

**PyTorch** ortalamanın biraz üstünde 196 FPS verdi. Ultralytics'in iç optimizasyonları ve PyTorch CUDA kernel'lerinin olgunluğu burada öne çıkıyor.

### Kaynak Kullanımı

Üç format da CPU kullanımında birbirine yakın (~%14). ONNX'in CPU max değerinin yüksek çıkması (%73.8), ONNX Runtime'ın bazı operasyonları CPU'da tamamlamasından kaynaklanıyor. TensorRT en düşük GPU utilization ile (%17.7) en yüksek FPS'i verdi — bu, kernel'lerin son derece verimli çalıştığının göstergesi.

### Doğruluk

Her üç format da pratik açıdan aynı sonuçları üretiyor. ~1.5'lik tespit farkı; frame başına ortalama 0.16 detection demek. Kritik güvenlik veya tıbbi uygulamalar için FP32 export tercih edilebilir; standart video analitik için FP16 yeterli.

---

## 🎯 Hangi Formatı Ne Zaman Kullanmalıyım?

| Senaryo | Önerilen Format |
|---------|----------------|
| Model eğitimi / araştırma | **PT** |
| Cross-platform deployment | **ONNX** |
| Mobil / edge cihazlar | **ONNX** (ONNX Runtime Mobile) |
| NVIDIA GPU'da production | **TensorRT** |
| Gerçek zamanlı video analitik | **TensorRT** |
| CPU-only sunucu | **ONNX** |
| Maksimum FPS (güvenlik kamerası, endüstriyel) | **TensorRT FP16** |
| Modeli farklı framework'e taşıma | **ONNX** |

---

## 🐛 Bilinen Sorunlar & Çözümler

### ONNX GPU — `CUDAExecutionProvider not available`

**Semptom:** ONNX çalışırken şu uyarı çıkar:
```
WARNING ⚠️ CUDA requested but CUDAExecutionProvider not available. Using CPU...
```

**Kök neden:** `torch-2.x+cu130` (CUDA 13.0) kuruluyken `onnxruntime-gpu` yükleniyor.  
`onnxruntime-gpu`, resmi olarak yalnızca **CUDA 12.x**'i destekler.
PyTorch'un CUDA 13 DLL'lerini ORT bulamıyor.

**Çözüm 1 — PyTorch'u CUDA 12'ye downgrade et (önerilen):**
```bash
pip uninstall torch torchvision torchaudio onnxruntime-gpu -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

**Çözüm 2 — ORT CUDA 13 nightly (deneysel):**
```bash
pip install flatbuffers numpy packaging protobuf sympy
pip install --pre --index-url \
  https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
  onnxruntime-gpu
```

**Doğrulama:**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# ['CUDAExecutionProvider', 'CPUExecutionProvider'] görünmeli
```

---

### TensorRT — `Invalid CUDA device=0`

**Semptom:**
```
torch.cuda.is_available(): False
torch.cuda.device_count(): 1
os.environ['CUDA_VISIBLE_DEVICES']: ''
```

**Kök neden:** `CUDA_VISIBLE_DEVICES` boş string olarak set edilmiş, bu PyTorch'un GPU'yu görmesini engelliyor.

**Çözüm:**
```bash
export CUDA_VISIBLE_DEVICES=0    # Linux
# veya Windows:
set CUDA_VISIBLE_DEVICES=0
```
`convert_models.py` bunu otomatik düzeltir, ama genel PyTorch scriptleri için manuel set gerekebilir.

---

### `.engine` dosyası başka makinede çalışmıyor

Normal — TensorRT engine'leri GPU mimarisine ve TensorRT versiyonuna özgüdür. Farklı bir makinede yeniden `convert_models.py` çalıştır.

> ⚠️ Bu benchmarkta engine build süresi **~142 saniye** oldu (RTX 4060 Laptop, TensorRT 10.16.1). Sunucu sınıfı GPU'larda bu süre daha kısa olabilir.

---

### ONNX opset hatası

`convert_models.py` içindeki `opset=12` değerini 17'ye yükselt veya 11'e düşür.

---

## 📁 Proje Yapısı

```
model-comparison/
├── scripts/
│   ├── convert_models.py    # PT → ONNX → TensorRT
│   ├── benchmark.py          # Video üzerinde test + metrikler
│   └── plot_results.py       # 3 karşılaştırma grafiği
├── models/
│   └── best.pt               # Senin modelin
├── outputs/                  # Annotasyonlu çıktı videoları
├── results/
│   ├── benchmark.json
│   ├── comparison_performance.png
│   ├── comparison_resources.png
│   └── comparison_accuracy.png
├── requirements.txt
├── 
└── README.md
```

---

## 📚 Kaynaklar

- [Ultralytics Export Docs](https://docs.ultralytics.com/modes/export/)
- [ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [onnxruntime-gpu Install](https://onnxruntime.ai/docs/install/)

---