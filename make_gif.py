import sys
import glob
import os
from PIL import Image


def make_gif(path, output="output.gif", duration=200):
    pattern = os.path.join(path, "iter*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No iter*.png files found in %s" % path)
        sys.exit(1)

    frames = [Image.open(f) for f in files]
    frames[0].save(
        os.path.join(path, output),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print("Saved %s (%d frames)" % (os.path.join(path, output), len(frames)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_gif.py <path> [output.gif] [duration_ms]")
        sys.exit(1)

    path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "output.gif"
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    make_gif(path, output, duration)
