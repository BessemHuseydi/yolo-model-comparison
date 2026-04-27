# YOLO Modelinizi Production'a Taşımadan Önce Bilmeniz Gereken 3 Format: PT, ONNX ve TensorRT Karşılaştırması

**Yayın tarihi:** 22 Nisan 2026  
**Kategori:** Teknik, Yapay Zeka, Bilgisayarlı Görü  
**Okuma süresi:** ~7 dakika

---

Bir nesne tespiti modeli eğittiniz, doğruluk metrikleriniz harika. Ama projeyi production ortamına aldığınızda fark ettiniz ki model beklediğiniz kadar hızlı değil. Belki de kullandığınız model formatı yanlış.

Biz de tam bu soruyu cevaplamak için YOLO modelimizi üç farklı formatta — `.pt`, `.onnx` ve `.engine` — çalıştırıp gerçek video üzerinde ölçtük. Sonuçlar bazı açılardan beklediğimizden farklı çıktı.

---

## Model Formatları Neden Önemli?

Derin öğrenme modellerini farklı ortamlarda çalıştırmak için tasarlanmış birden fazla format var. Her format farklı bir amaca hizmet ediyor:

**PyTorch `.pt`** — Araştırma ve geliştirme ortamının doğal formatı. Modeli eğittiğinizde zaten bu formatta elde ediyorsunuz.

**ONNX `.onnx`** — Open Neural Network Exchange. Framework'ten bağımsız bir ara format. PyTorch'ta eğittiğiniz modeli ONNX Runtime, TensorFlow veya OpenVINO gibi farklı motorlarda çalıştırmanıza olanak sağlıyor.

**TensorRT `.engine`** — NVIDIA'nın inference optimizer'ı. Modeli hedef GPU mimarisine özel olarak derliyor, layer fusion ve precision calibration gibi tekniklerle GPU'yu maksimum verimde kullanıyor.

Teoride TensorRT her zaman en hızlı, PT her zaman en yavaş. Ama rakamlar ne diyor?

---

## Test Ortamımız

- **GPU:** NVIDIA GeForce RTX 4060 Laptop
- **PyTorch:** 2.5.1+cu121
- **ONNX Runtime:** 1.23.2 (CUDAExecutionProvider ✅)
- **TensorRT:** 10.16.1.11
- **Model:** YOLOv8 object detection (6.3 MB `.pt`)
- **Test videosu:** 3.954 frame

---

## Sonuçlar

### Hız

| Format | Ortalama (ms) | p95 (ms) | FPS |
|--------|:-------------:|:--------:|:---:|
| PT | 5,10 | 6,97 | 196 |
| ONNX | 5,63 | 7,13 | 178 |
| **TensorRT** | **3,62** | **5,20** | **276** |

TensorRT beklediğimiz gibi açık ara önde: PT'ye göre **1,41×**, ONNX'e göre **1,55×** daha hızlı. 276 FPS, saniyede 276 kare işlemek demek — gerçek zamanlı güvenlik kamerası, üretim hattı kontrolü veya trafik analizi gibi uygulamalar için fazlasıyla yeterli.

Ama asıl ilginç bulgu şu: ONNX beklediğimizden daha iyi performans gösterdi. GPU üzerinde çalıştığında PT ile aradaki fark yalnızca %9. Bu, ONNX'in CPU'ya düştüğü eski CUDA uyumsuzluk senaryolarından çok farklı bir tablo.

### Kaynak Kullanımı

| Format | CPU Ort. | GPU Ort. | GPU Bellek |
|--------|:--------:|:--------:|:----------:|
| PT | %13,3 | %22,2 | 790 MB |
| ONNX | %14,0 | %30,1 | 926 MB |
| TensorRT | %14,6 | **%17,7** | 838 MB |

Burada TensorRT hakkında çarpıcı bir bulgu var: En düşük GPU kullanımıyla (%17,7) en yüksek FPS'i veriyor. CUDA kernel'lerinin ne kadar optimize edildiğinin somut bir göstergesi bu. Aynı makine üzerinde başka servisler veya modeller de çalıştırıyorsanız TensorRT'nin bu "verimli" yapısı büyük avantaj sağlıyor.

### Doğruluk

| Format | Toplam Tespit | Ort. Güven | Frame Eşleşme | Fark |
|--------|:-------------:|:----------:|:-------------:|:----:|
| PT | 22.199 | 0,789 | referans | — |
| ONNX | 21.865 | 0,794 | %85,3 | −1,5% |
| TensorRT | 21.870 | 0,794 | %85,5 | −1,5% |

TensorRT FP16 modunda çalışmasına karşın PT ile neredeyse özdeş sonuçlar üretiyor. Frame eşleşme oranı %85'in üzerinde; toplam tespit farkı ise sadece %1,5. Pratikte bu, frame başına ortalama 0,16 adet fazla ya da eksik tespit anlamına geliyor. Gerçek dünya nesne tespiti uygulamalarında bu fark genellikle ihmal edilebilir düzeyde.

---

## Ne Zaman Hangi Formatı Kullanmalısınız?

Sonuçlar iyi yorumlanmadığında yanlış format seçimi ciddi performans kayıplarına yol açabiliyor. Pratik bir rehber:

**PyTorch `.pt` seçin** eğer hâlâ model geliştirme veya fine-tuning aşamasındaysanız. Debug kolaylığı ve Python ekosistemiyle tam entegrasyon araştırma ortamı için biçilmiş kaftan.

**ONNX `.onnx` seçin** eğer modeli birden fazla platformda veya farklı donanımlarda çalıştırmanız gerekiyorsa. AMD GPU, Intel NPU, edge cihazlar veya tarayıcı (ONNX.js) hedefliyorsanız ONNX'in taşınabilirliği rakipsiz. Ayrıca saf CPU ortamlarında da ONNX Runtime genellikle PyTorch'tan daha hızlı inference yapıyor.

**TensorRT `.engine` seçin** eğer NVIDIA GPU'da maksimum hızı istiyorsanız ve production ortamınız sabitse. Üretim hattı görüntü analizi, güvenlik kamerası, otonom sistem gibi latency-critical uygulamalarda TensorRT vazgeçilmez.

---

## Dikkat Edilmesi Gereken Bir Nokta: CUDA Uyumluluğu

TensorRT engine dosyaları GPU mimarisine özgüdür. Farklı bir sunucuya veya farklı bir NVIDIA GPU'ya geçtiğinizde engine'i yeniden build etmeniz gerekiyor. Bu testte build süresi yaklaşık 142 saniye oldu — production deployment planlamanıza dahil etmeniz gereken bir maliyet.

ONNX tarafında ise dikkat edilmesi gereken şey CUDA sürümü uyumluluğu. `onnxruntime-gpu` resmi olarak CUDA 12.x'i destekliyor. CUDA 13 kullanan ortamlarda `CUDAExecutionProvider` devreye girmeyebilir ve model sessizce CPU'ya düşebilir. Bu durumda ONNX GPU potansiyelinin çok altında bir performans elde edersiniz. Kurulumunuzu şu komutla doğrulayabilirsiniz:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Beklenen: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## Sonuç

Üç format arasındaki hız farkı yalnızca "daha hızlı çalışmak" meselesinden ibaret değil. Aynı zamanda kaynak kullanımı, bakım maliyeti ve deployment esnekliği açısından da önemli trade-off'lar içeriyor.

Bu benchmark bize şunu söylüyor: Production NVIDIA GPU ortamlarında TensorRT, hem hız hem kaynak verimliliği açısından belirgin biçimde öne çıkıyor. Ama çoklu platform desteği veya hızlı iterasyon önceliğinizse ONNX güçlü bir alternatif.

Model formatı seçimi, mimari karar kadar önemli. Projenizin ihtiyaçlarını ve altyapınızı göz önünde bulundurarak doğru formatı seçmek, production performansını doğrudan etkiliyor.

---

*Bu benchmark RTX 4060 Laptop GPU üzerinde gerçekleştirilmiştir. Farklı GPU modellerinde oranlar değişkenlik gösterebilir, ancak genel eğilimler tutarlılığını korur.*

---

**Etiketler:** YOLO, TensorRT, ONNX, PyTorch, Nesne Tespiti, Bilgisayarlı Görü, Model Optimizasyonu, Production AI
