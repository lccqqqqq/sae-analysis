"""Download the WikiText-2 dataset to the repo root as plain-text files.

Usage:
    python scripts/utils/download_data.py

Writes:
    wikitext-2-train.txt  (~10 MB)
    wikitext-2-test.txt   (~1.2 MB)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def download():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install 'datasets': pip install datasets")

    print("Downloading WikiText-2 …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    for split, fname in [("train", "wikitext-2-train.txt"), ("test", "wikitext-2-test.txt")]:
        out = ROOT / fname
        with open(out, "w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(row["text"])
        size_mb = out.stat().st_size / 1e6
        print(f"  Wrote {fname} ({size_mb:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    download()
