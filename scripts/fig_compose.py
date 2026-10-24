"""Compose multi-panel manuscript figures from single-panel PNGs (adds (a)/(b) labels)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image, ImageDraw, ImageFont
from cdrt import config as C


def stack(paths, out, labels=("(a)", "(b)", "(c)")):
    ims = [Image.open(C.FIGURES / p).convert("RGB") for p in paths]
    W = max(i.width for i in ims)
    ims = [i.resize((W, int(i.height * W / i.width)), Image.LANCZOS) if i.width != W else i for i in ims]
    pad = int(0.02 * W)
    H = sum(i.height for i in ims) + pad * (len(ims) - 1)
    lm = int(0.035 * W)                      # left margin so panel letters never cover axis labels
    canvas = Image.new("RGB", (W + lm, H), "white")
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(0.018 * W))
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for im, lab in zip(ims, labels):
        canvas.paste(im, (lm, y))
        d.text((int(0.002 * W), y + int(0.002 * W)), lab, fill="black", font=font)
        y += im.height + pad
    canvas.save(C.FIGURES / out, dpi=(600, 600))


if __name__ == "__main__":
    stack(["fig2d_measured_vs_clean.png", "fig2e_fast_model_validation.png"], "fig4_measurement.png")
    print("composed")
