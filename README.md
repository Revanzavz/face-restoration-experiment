# Region-Aware Old-Photo Restoration

Ekstensi untuk project **Old-Photo-Restoration** (U-Net restorasi global) menjadi
pipeline region-aware: deteksi wajah, restorasi wajah terspesialisasi, lalu
blend kembali ke citra global. Wajah diberi model khusus (GFPGAN / CodeFormer),
sementara sisa citra ditangani U-Net global.

## Pipeline

```
citra rusak
  1. restorasi GLOBAL .............. U-Net (best_unet.pth)
  2. DETEKSI wajah ................. RetinaFace -> bbox + 5 landmark
  3. per wajah:
       a. ALIGN ke 512x512 ......... similarity transform (5 landmark)
       b. RESTORASI wajah .......... GFPGAN / CodeFormer
       c. SEGMENTASI mask .......... BiSeNet face parsing (opsional)
       d. warp balik + BLEND ....... feather (alpha) / Poisson (seamlessClone)
  -> hasil region-aware
```

## Struktur folder

```
face-restoration-experiment/
├── face_restore/
│   ├── detector.py
│   ├── aligner.py
│   ├── parsing.py
│   ├── face_restorer.py
│   ├── blending.py
│   ├── global_restore.py
│   ├── pipeline.py
│   ├── metrics.py
│   └── utils.py
├── experiments/
│   └── region_aware_restoration.ipynb
├── scripts/
│   ├── download_weights.py
│   ├── download_ffhq.py
│   └── patch_basicsr.py
├── weights/
├── data/clean_faces/
├── results/
├── requirements_face.txt
└── README.md
```

Letakkan folder ini di samping folder `Old-Photo-Restoration` (yang berisi
`model.py` dan `best_unet.pth`). Notebook mengasumsikan struktur:
`…/Old-Photo-Restoration/` dan `…/face-restoration-experiment/` bersebelahan.

## Instalasi

```bash
pip install -r requirements_face.txt

python scripts/patch_basicsr.py

python scripts/download_weights.py
```

`patch_basicsr.py` diperlukan karena torchvision >= 0.17 menghapus
`torchvision.transforms.functional_tensor` yang masih diimpor basicsr.
Script mengganti satu baris import agar `import gfpgan` tidak gagal.

## Penggunaan

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
utils.imwrite_bgr("hasil.png", res.region_aware)
```

## Eksperimen

| # | Eksperimen | Output |
|---|------------|--------|
| 1 | Deteksi RetinaFace (bbox + landmark) | visualisasi |
| 2 | Alignment wajah ke 512x512 | crop teraligned |
| 3 | GFPGAN vs CodeFormer pada crop sama | grid perbandingan |
| 4 | Pengaruh fidelity `w` CodeFormer (0.0 / 0.5 / 0.9) | grid |
| 5 | Segmentasi face parsing | mask + overlay |
| 6 | Baseline (global-only) vs region-aware | grid |
| 7 | Strategi blending: feather vs Poisson | perbandingan |
| 8 | Evaluasi kuantitatif: PSNR / SSIM / LPIPS | tabel + bar chart |

Evaluasi kuantitatif (Bagian 8) menggunakan wajah bersih dari `data/clean_faces/`
yang didegradasi secara sintetik (`utils.synthesize_degradation`), lalu
dibandingkan ke versi bersih. Untuk foto asli tanpa ground-truth, pakai
`metrics.sharpness` sebagai metrik no-reference.

## Pilihan desain

- `face_source='global'` (default): deteksi dan restorasi dilakukan pada hasil
  U-Net global. Set ke `'original'` agar GFPGAN/CodeFormer melihat degradasi mentah.
- Restorasi memanggil jaringan langsung pada crop teraligned, bukan API satu-baris
  GFPGANer, supaya tiap tahap eksplisit dan bisa dianalisis.
- `MTCNNDetector` tersedia di `detector.py` sebagai alternatif RetinaFace
  (butuh `facenet-pytorch`).

## Referensi

- RetinaFace: Deng et al., *RetinaFace: Single-stage Dense Face Localisation*, 2019.
- GFPGAN: Wang et al., *Towards Real-World Blind Face Restoration with Generative Facial Prior*, CVPR 2021.
- CodeFormer: Zhou et al., *Towards Robust Blind Face Restoration with Codebook Lookup Transformer*, NeurIPS 2022.
- BiSeNet face parsing via facexlib.
- LPIPS: Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018.
