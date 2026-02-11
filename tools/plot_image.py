#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Plot z-slice from image.bin")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--z", type=int, default=48)
    p.add_argument("--input", type=str, default="image.bin")
    p.add_argument("--output", type=str, default="image_slice.png")
    args = p.parse_args()

    arr = np.fromfile(args.input, dtype=np.float32)
    vol = arr.reshape(args.nz, args.ny, args.nx)
    sl = vol[args.z]

    plt.figure(figsize=(8, 6))
    vmax = np.percentile(np.abs(sl), 99)
    plt.imshow(sl, cmap="seismic", vmin=-vmax, vmax=vmax, origin="lower")
    plt.colorbar(label="RTM amplitude")
    plt.title(f"RTM image z={args.z}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
