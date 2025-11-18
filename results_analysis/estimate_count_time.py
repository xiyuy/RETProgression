#!/usr/bin/env python3
import os, time, random, argparse
from PIL import Image, UnidentifiedImageError

EXTS = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')

def iter_images(root):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                yield os.path.join(dp, fn)

def hms(sec):
    sec = int(sec); h = sec//3600; m = (sec%3600)//60; s = sec%60
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    ap = argparse.ArgumentParser(description="Benchmark image-size scanning and estimate total time.")
    ap.add_argument("img_dir")
    ap.add_argument("--sample", type=int, default=2000, help="files to benchmark (default 2000)")
    args = ap.parse_args()

    # enumerate files
    t0 = time.perf_counter()
    paths = list(iter_images(args.img_dir))
    total = len(paths)
    enum_dt = time.perf_counter() - t0
    if total == 0:
        print("No images found."); return

    # sample and benchmark
    sample_n = min(args.sample, total)
    random.shuffle(paths)
    bench = paths[:sample_n]

    t1 = time.perf_counter()
    ok = 0
    for p in bench:
        try:
            with Image.open(p) as im:
                _ = im.size  # header-only read
            ok += 1
        except (UnidentifiedImageError, OSError):
            pass
    dt = time.perf_counter() - t1
    rate = ok / dt if dt > 0 else 0.0
    eta = (total / rate) if rate > 0 else float("inf")

    print(f"Found {total:,} images (enumeration {enum_dt:.2f}s).")
    print(f"Benchmark: {ok}/{sample_n} in {dt:.2f}s  →  {rate:.1f} images/sec")
    print(f"Estimated time to scan all: ~{hms(eta)} at this rate.")

if __name__ == "__main__":
    main()
