# Vision Transformer Comparison

Proyek penelitian yang membandingkan performa tiga arsitektur Vision Transformer state-of-the-art: **ViT (Vision Transformer)**, **DeiT (Data-efficient Image Transformer)**, dan **Swin Transformer** pada dataset STL-10.

📚 **Mata Kuliah**: Pembelajaran Mendalam (IF25-40401)  
👨‍🏫 **Dosen**: Imam Eko Wicaksono, S.Si., M.Si. & Martin Clinton Tosima Manullang, S.T., M.T., Ph.D.

## 🎯 Tujuan Penelitian

Penelitian ini bertujuan untuk:

1. Membandingkan performa akurasi dari tiga arsitektur Vision Transformer (ViT, DeiT, Swin)
2. Menganalisis efisiensi komputasi (training time, inference time, throughput)
3. Mengevaluasi trade-off antara akurasi dan efisiensi untuk aplikasi praktis
4. Memberikan rekomendasi pemilihan model berdasarkan skenario penggunaan

## 📊 Dataset

**STL-10 Dataset**

- **Jumlah Kelas**: 10 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- **Resolusi Input**: 224 × 224 piksel (diubah dari asli 96 × 96)
- **Pembagian Data**:
  - Training: 4,000 gambar
  - Validation: 1,000 gambar
  - Test: 8,000 gambar

**Augmentasi Data**:

- Random Horizontal Flip (50%)
- Random Rotation (±15°)
- Color Jitter (brightness & contrast: 0.2)
- Normalisasi ImageNet (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

## 🏗️ Arsitektur Model

| Model         | Parameter | Ukuran | Karakteristik Utama                     |
| ------------- | --------- | ------ | --------------------------------------- |
| **ViT Base**  | 85.8M     | 327 MB | Pure transformer, patch-based attention |
| **DeiT Base** | 85.8M     | 327 MB | Knowledge distillation, data-efficient  |
| **Swin Base** | 86.8M     | 332 MB | Hierarchical, shifted window attention  |

Semua model menggunakan **pre-trained weights** dari ImageNet dan di-fine-tune pada STL-10.

## 🔧 Konfigurasi Training

```python
Optimizer          : AdamW
Learning Rate      : 3e-5
Batch Size         : 64
Epochs             : 20
Loss Function      : Cross Entropy Loss
LR Scheduler       : Cosine Annealing (T_max=20)
Weight Decay       : 0.05
Mixed Precision    : Enabled (torch.cuda.amp)
```

**Hardware**:

- GPU: NVIDIA A40 (44 GB VRAM)
- CPU: 9 vCores
- RAM: 50 GB
- Platform: RunPod Cloud

## 📈 Hasil Eksperimen

### Ringkasan Performa

| Model         | Test Accuracy | Macro F1   | Training Time | Inference/Img | Throughput  |
| ------------- | ------------- | ---------- | ------------- | ------------- | ----------- |
| **DeiT Base** | 98.11%        | 0.9811     | 5.5 menit     | 2.56 ms       | 389.9 img/s |
| **ViT Base**  | 99.08%        | 0.9907     | 5.5 menit     | 2.57 ms       | 388.8 img/s |
| **Swin Base** | **99.36%**    | **0.9936** | 8.5 menit     | 3.42 ms       | 292.6 img/s |

### Kesimpulan Utama

✅ **Swin Transformer**: Akurasi tertinggi (99.36%), cocok untuk aplikasi yang memprioritaskan akurasi  
✅ **ViT Base**: Keseimbangan terbaik antara akurasi (99.08%) dan kecepatan inferensi  
✅ **DeiT Base**: Tercepat dalam inferensi (389.9 img/s), cocok untuk real-time applications

## 📁 Struktur Proyek

```
VisionTransformer-Comparison/
├── chapters/                    # Bab-bab Laporan LaTeX
│   ├── cover.tex               # Halaman cover
│   ├── 01_pendahuluan.tex      # BAB I: Pendahuluan
│   ├── 02_landasan_teori.tex   # BAB II: Landasan Teori
│   ├── 03_metodologi.tex       # BAB III: Metodologi
│   ├── 04_hasil_analisis.tex   # BAB IV: Hasil dan Analisis
│   ├── 05_kesimpulan.tex       # BAB V: Kesimpulan dan Saran
│   └── 06_lampiran.tex         # LAMPIRAN
├── code/                        # Kode Eksperimen
│   ├── 122140055_VisionTransformerComparison.ipynb  # Notebook utama
│   ├── requirements.txt         # Dependencies
│   ├── DeiT_Base_history.csv   # Training log DeiT
│   ├── ViT_Base_history.csv    # Training log ViT
│   ├── Swin_Base_history.csv   # Training log Swin
│   ├── final_summary.csv       # Ringkasan hasil
│   └── detailed_metrics.txt    # Metrik detail per kelas
├── Figure/                      # Visualisasi dan Gambar
│   ├── *_Accuracy_Trend.pdf    # Grafik akurasi
│   ├── *_Loss_Trend.pdf        # Grafik loss
│   ├── *_Confusion_Matrix.pdf  # Confusion matrices
│   ├── learning_curves.pdf     # Learning curves
│   ├── training_log*.png       # Screenshot training logs
│   └── *.pdf                   # Diagram arsitektur
├── main.tex                     # Dokumen LaTeX utama
├── Referensi.bib               # File bibliografi
└── Readme.MD                   # File ini
```

## 🚀 Cara Menjalankan

### 1. Setup Environment

```bash
cd code/
pip install -r requirements.txt
```

### 2. Jalankan Eksperimen

Buka dan jalankan notebook:

```bash
jupyter notebook 122140055_VisionTransformerComparison.ipynb
```

Atau jalankan di Google Colab/RunPod untuk akses GPU.

### 3. Kompilasi Dokumen LaTeX

```bash
# Dari root directory
pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
```

**Note**: Flag `-shell-escape` diperlukan untuk package `minted` dan `svg`.

## 📦 Dependencies

**Python Libraries**:

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm (PyTorch Image Models)
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

**LaTeX Packages**:

- minted (syntax highlighting)
- svg (SVG image support)
- hyperref, graphicx, amsmath, booktabs, dll.

Lihat `code/requirements.txt` untuk daftar lengkap dependencies Python.

## 📄 Metrik Evaluasi

1. **Accuracy**: Persentase prediksi benar
2. **Precision, Recall, F1-Score**: Per-class performance
3. **Confusion Matrix**: Visualisasi prediksi vs ground truth
4. **Training Time**: Waktu total training (20 epochs)
5. **Inference Time**: Rata-rata waktu per gambar (ms)
6. **Throughput**: Jumlah gambar diproses per detik
7. **Model Size**: Total parameters dan ukuran file (MB)

## 📚 Referensi Utama

1. Dosovitskiy et al. (2020) - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
2. Touvron et al. (2021) - Training Data-Efficient Image Transformers
3. Liu et al. (2021) - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
4. Vaswani et al. (2017) - Attention is All You Need
