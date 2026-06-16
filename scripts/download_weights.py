import os
import urllib.request

WEIGHTS = {
    "GFPGANv1.4.pth": (
        "https://github.com/TencentARC/GFPGAN/releases/download/"
        "v1.3.4/GFPGANv1.4.pth"
    ),
    "codeformer.pth": (
        "https://github.com/sczhou/CodeFormer/releases/download/"
        "v0.1.0/codeformer.pth"
    ),
}

DEST = os.path.join(os.path.dirname(__file__), "..", "weights")


def _progress(blocks, bs, total):
    if total > 0:
        pct = min(100, blocks * bs * 100 // total)
        print(f"\r  {pct:3d}%", end="", flush=True)


def main():
    os.makedirs(DEST, exist_ok=True)
    for name, url in WEIGHTS.items():
        out = os.path.join(DEST, name)
        if os.path.exists(out):
            print(f"[skip] {name} sudah ada")
            continue
        print(f"[unduh] {name}\n  {url}")
        urllib.request.urlretrieve(url, out, _progress)
        print("  selesai")
    print(f"\nBobot tersimpan di: {os.path.abspath(DEST)}")


if __name__ == "__main__":
    main()
