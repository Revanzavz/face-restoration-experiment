import importlib.util
import os
import re

OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW = "from torchvision.transforms.functional import rgb_to_grayscale"


def find_degradations_py() -> str | None:
    spec = importlib.util.find_spec("basicsr")
    if spec is None or not spec.submodule_search_locations:
        return None
    base = list(spec.submodule_search_locations)[0]
    path = os.path.join(base, "data", "degradations.py")
    return path if os.path.exists(path) else None


def main():
    path = find_degradations_py()
    if path is None:
        print("basicsr belum terpasang / file degradations.py tidak ditemukan.")
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if OLD in content:
        content = content.replace(OLD, NEW)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[patched] {path}")
    elif NEW in content:
        print("[ok] sudah ter-patch sebelumnya")
    else:
        patched = re.sub(
            r"from\s+torchvision\.transforms\.functional_tensor\s+import\s+rgb_to_grayscale",
            NEW,
            content,
        )
        if patched != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)
            print(f"[patched-regex] {path}")
        else:
            print("[info] baris import target tidak ditemukan; mungkin versi berbeda.")


if __name__ == "__main__":
    main()
