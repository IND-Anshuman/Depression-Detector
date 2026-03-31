from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


MODELS = {
    "5": {
        "url": "https://raw.githubusercontent.com/face-analysis/emonet/master/pretrained/emonet_5.pth",
        "sha256": "1d8fac689dc04fc65a8a25c050bd5888e1fc794fcf71c2b20a1f2e6b78d933dd",
        "filename": "emonet_5.pth",
    },
    "8": {
        "url": "https://raw.githubusercontent.com/face-analysis/emonet/master/pretrained/emonet_8.pth",
        "sha256": "52918cffba56f31886e6959f6837266a5f7ef5c0a552baf4c6dabe1e8fa9bc97",
        "filename": "emonet_8.pth",
    },
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    with urlopen(url) as response, dest.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download official EmoNet checkpoints with explicit integrity checks.")
    ap.add_argument("--variant", choices=sorted(MODELS.keys()), default="8")
    ap.add_argument("--dest-dir", default="artifacts/models/emonet")
    args = ap.parse_args()

    spec = MODELS[args.variant]
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec["filename"]

    print(f"Downloading {spec['url']} -> {dest}")
    download(spec["url"], dest)
    actual = sha256_of(dest)
    if actual.lower() != spec["sha256"]:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {dest.name}: expected {spec['sha256']} got {actual}. "
            "Download was removed to avoid silent corruption."
        )
    print(f"Saved {dest} (sha256={actual})")


if __name__ == "__main__":
    main()
