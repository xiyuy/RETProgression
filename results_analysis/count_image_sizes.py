#!/usr/bin/env python3
import os
import sys
import time
from collections import Counter
from argparse import ArgumentParser
from PIL import Image, UnidentifiedImageError

def iter_images(root, exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn.lower())[1] in exts:
                yield os.path.join(dp, fn)

def main():
    ap = ArgumentParser(description="Count image sizes (WxH) in a directory.")
    ap.add_argument("img_dir", help="Directory containing images")
    ap.add_argument("--csv", help="Optional output CSV for size counts")
    ap.add_argument("--bad", help="Optional output text file for unreadable images")
    ap.add_argument("--every", type=int, default=10000,
                    help="Print progress every N images (default: 10000)")
    args = ap.parse_args()

    if not os.path.isdir(args.img_dir):
        print(f"Not a directory: {args.img_dir}", file=sys.stderr)
        sys.exit(1)

    sizes = Counter()
    bad = []

    total_seen = 0
    t0 = time.time()

    for path in iter_images(args.img_dir):
        total_seen += 1
        try:
            with Image.open(path) as im:
                w, h = im.size
            sizes[(w, h)] += 1
        except (UnidentifiedImageError, OSError) as e:
            bad.append((path, str(e)))

        if args.every > 0 and total_seen % args.every == 0:
            dt = time.time() - t0
            rate = total_seen / dt if dt > 0 else 0.0
            print(f"[progress] Processed {total_seen:,} images "
                  f"(~{rate:.1f} img/s)", flush=True)

    # Final summary
    total_ok = sum(sizes.values())
    print(f"\nScanned: {args.img_dir}")
    print(f"Readable images: {total_ok:,}")
    print(f"Unreadable images: {len(bad):,}")
    print("\nSize counts (sorted by frequency):")
    for (w, h), cnt in sizes.most_common():
        pct = 100.0 * cnt / total_ok if total_ok else 0.0
        print(f"  {w}x{h}: {cnt:,} ({pct:.2f}%)")

    # Optional CSV
    if args.csv:
        try:
            import csv
            with open(args.csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["width", "height", "count"])
                for (w, h), cnt in sorted(sizes.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
                    writer.writerow([w, h, cnt])
            print(f"\nWrote size counts to {args.csv}")
        except Exception as e:
            print(f"Could not write CSV: {e}", file=sys.stderr)

    # Optional bad files list
    if args.bad and bad:
        try:
            with open(args.bad, "w") as f:
                for p, err in bad:
                    f.write(f"{p}\t{err}\n")
            print(f"Wrote unreadable file list to {args.bad}")
        except Exception as e:
            print(f"Could not write bad list: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

