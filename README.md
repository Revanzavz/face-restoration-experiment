# Region-Aware Old-Photo Restoration

Ekstensi untuk project **Old-Photo-Restoration** (U-Net restorasi global) menjadi
pipeline *region-aware*: **deteksi wajah → restorasi wajah terspesialisasi → blend
kembali ke citra global**. Foto lama hampir selalu berisi wajah, dan wajah adalah
region paling diperhatikan mata — jadi region itu diberi model khusus
(GFPGAN / CodeFormer), sementara sisa citra ditangani U-Net global.

Ini problem CV sejati karena menggabungkan **deteksi + segmentasi + pemrosesan
kondisional**.

## Pipeline

```
citra rusak
  1. restorasi GLOBAL .............. U-Net (best_unet.pth, project asli)
  2. DETEKSI wajah ................. RetinaFace  -> bbox + 5 landmark
  3. per wajah:
       a. ALIGN ke 512x512 ......... similarity transform (5 landmark)
       b. RESTORASI wajah .......... GFPGAN  /  CodeFormer
       c. SEGMENTASI mask .......... BiSeNet face parsing  (opsional)
       d. warp balik + BLEND ....... feather (alpha) / Poisson (seamlessClone)
  -> hasil region-aware
```

## Struktur folder

```
face-restoration-experiment/
├── face_restore/                # package modul
│   ├── detector.py              # RetinaFace (+ MTCNN opsional)
│   ├── aligner.py               # alignment 5-landmark ke template FFHQ 512
│   ├── parsing.py               # face parsing / segmentasi (BiSeNet)
│   ├── face_restorer.py         # GFPGAN & CodeFormer (pada crop teraligned)
│   ├── blending.py              # feather & Poisson blending
│   ├── global_restore.py        # wrapper U-Net global (model.py existing)
│   ├── pipeline.py              # orkestrator RegionAwareRestorer
│   ├── metrics.py               # PSNR / SSIM / LPIPS + sharpness
│   └── utils.py                 # degradasi sintetik + grid visual + I/O
├── experiments/
│   └── region_aware_restoration.ipynb   # notebook eksperimen lengkap
├── scripts/
│   ├── download_weights.py      # unduh bobot GFPGAN & CodeFormer
│   └── patch_basicsr.py         # fix kompatibilitas torchvision
├── weights/                     # bobot model (terisi setelah download)
├── data/clean_faces/            # (isi sendiri) wajah bersih utk evaluasi kuantitatif
├── results/                     # output citra + tabel metrik
├── requirements_face.txt
└── README.md
```

> Letakkan folder ini **di samping** folder `Old-Photo-Restoration` (yang berisi
> `model.py` dan `best_unet.pth`). Notebook mengasumsikan struktur:
> `…/Old-Photo-Restoration/` dan `…/face-restoration-experiment/` bersebelahan.

## Instalasi

```bash
# 1. aktifkan environment project Old-Photo-Restoration (torch sudah terpasang di sana)
# 2. install dependensi tambahan
pip install -r requirements_face.txt

# 3. PENTING: fix kompatibilitas basicsr <-> torchvision baru (sekali saja)
python scripts/patch_basicsr.py

# 4. unduh bobot GFPGAN & CodeFormer (RetinaFace & BiSeNet otomatis oleh facexlib)
python scripts/download_weights.py
```

### Kenapa perlu `patch_basicsr.py`?
torchvision ≥ 0.17 (project ini memakai 0.22) menghapus
`torchvision.transforms.functional_tensor`, yang masih diimpor `basicsr`. Tanpa
patch, `import gfpgan` akan gagal. Script tinggal mengganti satu baris import.

## Menjalankan

### Notebook (untuk laporan / eksperimen)
Buka `experiments/region_aware_restoration.ipynb`. Isinya runtut: deteksi →
alignment → restorasi (GFPGAN vs CodeFormer) → segmentasi → pipeline penuh →
perbandingan baseline vs region-aware → evaluasi kuantitatif → simpan hasil.

### Pakai dari kode
```python
from face_restore.global_restore import GlobalRestorer
from face_restore.detector import RetinaFaceDetector
from face_restore.face_restorer import GFPGANRestorer
from face_restore.parsing import FaceParser
from face_restore.pipeline import RegionAwareRestorer
from face_restore import utils

gr  = GlobalRestorer("../Old-Photo-Restoration/best_unet.pth", device="cuda")
gfp = GFPGANRestorer("weights/GFPGANv1.4.pth", device="cuda")
det = RetinaFaceDetector(device="cuda")
par = FaceParser(device="cuda")

pipe = RegionAwareRestorer(gr, gfp, detector=det, parser=par,
                           face_source="global", blend_method="feather")
res = pipe.restore(utils.imread_bgr("foto_lama.jpg"))
utils.imwrite_bgr("hasil.png", res.region_aware)   # vs res.global_only (baseline)
```

## Eksperimen yang disediakan

| # | Eksperimen | Output |
|---|------------|--------|
| 1 | Deteksi RetinaFace (bbox + landmark) | visualisasi |
| 2 | Alignment wajah ke 512×512 | crop teraligned |
| 3 | GFPGAN vs CodeFormer pada crop sama | grid perbandingan |
| 4 | Pengaruh fidelity `w` CodeFormer (0.0 / 0.5 / 0.9) | grid |
| 5 | Segmentasi face parsing | mask + overlay |
| 6 | Baseline (global-only) vs region-aware (GFPGAN / CodeFormer) | grid |
| 7 | Strategi blending: feather vs Poisson | perbandingan |
| 8 | **Kuantitatif**: degradasi sintetik → PSNR / SSIM / LPIPS | tabel + bar chart |

### Catatan evaluasi kuantitatif
Foto lama asli tidak punya ground-truth, jadi PSNR/SSIM/LPIPS tidak bisa dihitung
langsung. Solusi standar di *blind face restoration*: ambil **wajah bersih**
(mis. subset CelebA-HQ / FFHQ / foto pribadi tajam), terapkan **degradasi sintetik**
(`utils.synthesize_degradation`: blur → downscale → noise → JPEG), lalu restorasi
dan bandingkan ke versi bersih. Isi folder `data/clean_faces/` lalu jalankan
Bagian 8 notebook. Untuk foto asli tanpa GT, pakai `metrics.sharpness` (no-reference).

## Pilihan desain

- **`face_source='global'`** (default): deteksi & restorasi wajah dilakukan pada
  hasil U-Net global → deteksi lebih andal, lalu wajah disempurnakan. Set
  `'original'` bila ingin GFPGAN/CodeFormer melihat degradasi mentah.
- **Restorasi memanggil jaringan langsung** pada crop teraligned (bukan API
  satu-baris GFPGANer), supaya tiap tahap (deteksi, segmentasi, blending)
  eksplisit dan bisa dianalisis untuk laporan.
- **Detektor**: RetinaFace default; `MTCNNDetector` tersedia di `detector.py`
  untuk eksperimen perbandingan (butuh `facenet-pytorch`).

## Referensi
- RetinaFace: Deng et al., *RetinaFace: Single-stage Dense Face Localisation*, 2019.
- GFPGAN: Wang et al., *Towards Real-World Blind Face Restoration with Generative Facial Prior*, CVPR 2021.
- CodeFormer: Zhou et al., *Towards Robust Blind Face Restoration with Codebook Lookup Transformer*, NeurIPS 2022.
- BiSeNet (face parsing) via facexlib.
- LPIPS: Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018.
