import argparse
import io
import os
import time
import urllib.request

CANDIDATE_REPOS = [
    ("Dmini/FFHQ-512x512", None),
    ("bitmind/ffhq-256", None),
    ("nuwandaa/ffhq256", None),
    ("merkol/ffhq-256", None),
]


def _extract_image(example):
    for key in ("image", "img", "jpg", "png"):
        if key in example and example[key] is not None:
            return example[key]
    for v in example.values():
        if hasattr(v, "convert"):
            return v
    return None


def from_huggingface(num, size, out):
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[ffhq] paket 'datasets' tidak ada ({e}); lewati ke fallback.")
        return 0

    for repo, config in CANDIDATE_REPOS:
        try:
            print(f"[ffhq] coba {repo} (streaming) ...")
            ds = load_dataset(repo, config, split="train", streaming=True)
            saved = 0
            for ex in ds:
                img = _extract_image(ex)
                if img is None:
                    continue
                img.convert("RGB").resize((size, size)).save(
                    os.path.join(out, f"ffhq_{saved:04d}.png")
                )
                saved += 1
                print(f"\r  tersimpan {saved}/{num}", end="", flush=True)
                if saved >= num:
                    break
            print()
            if saved > 0:
                return saved
        except Exception as e:
            print(f"  gagal: {e}")
    return 0


def from_tpdne(num, size, out):
    from PIL import Image

    url = "https://thispersondoesnotexist.com"
    saved = 0
    print("[fallback] mengambil wajah dari thispersondoesnotexist.com ...")
    for i in range(num * 2):
        if saved >= num:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=20).read()
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size))
            img.save(os.path.join(out, f"face_{saved:04d}.png"))
            saved += 1
            print(f"\r  tersimpan {saved}/{num}", end="", flush=True)
            time.sleep(1.0)
        except Exception as e:
            print(f"\n  lewati satu ({e})")
            time.sleep(1.5)
    print()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--source", choices=["auto", "ffhq", "tpdne"], default="auto")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "clean_faces"),
    )
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    saved = 0
    if args.source in ("auto", "ffhq"):
        saved = from_huggingface(args.num, args.size, out)
    if saved == 0 and args.source in ("auto", "tpdne"):
        saved = from_tpdne(args.num, args.size, out)

    if saved > 0:
        print(f"\nSelesai: {saved} wajah bersih -> {out}")
    else:
        print(
            "\nGagal mengunduh dari semua sumber. Periksa koneksi internet, atau "
            "isi folder berikut secara manual dengan foto wajah bersih:\n  " + out
        )


if __name__ == "__main__":
    main()
