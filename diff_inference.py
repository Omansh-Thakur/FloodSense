import argparse
import subprocess
import os
import cv2
import numpy as np
import tifffile
import tempfile

def run_deepwater_inference(checkpoint, image_path, save_path):
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", checkpoint,
        "--image_path", image_path,
        "--save_path", save_path
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def to_uint8_rgb_from_6band(arr6):
    bands = []
    for b in (2,1,0):
        band = arr6[..., b].astype(float)
        p2, p98 = np.percentile(band, (2,98))
        band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
        bands.append((band * 255).astype(np.uint8))
    rgb = np.stack(bands[::-1], axis=-1)
    return rgb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "--checkpoint_path", dest="checkpoint", required=True)
    p.add_argument("--image1", required=True)
    p.add_argument("--image2", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--threshold", type=int, default=127, help="binary threshold for inference output images (0-255). If inference outputs grayscale probability image 0-255, threshold around 127. Adjust as needed.")
    p.add_argument("--morph", type=int, default=3, help="morph kernel size for postprocessing")
    args = p.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="dwm_diff_")
    out1 = os.path.join(tmpdir, "pred1.png")
    out2 = os.path.join(tmpdir, "pred2.png")

    # 1) Run inference on both images
    run_deepwater_inference(args.checkpoint, args.image1, out1)
    run_deepwater_inference(args.checkpoint, args.image2, out2)

    # 2) load predicted images as grayscale
    p1 = cv2.imread(out1, cv2.IMREAD_GRAYSCALE)
    p2 = cv2.imread(out2, cv2.IMREAD_GRAYSCALE)
    if p1 is None or p2 is None:
        raise RuntimeError("Could not read predictions. Check inference.py output paths.")

    if p1.shape != p2.shape:
        raise RuntimeError(f"Prediction outputs have different shapes: {p1.shape} vs {p2.shape}. Ensure input images are co-registered and same size.")

    # 3) threshold to binary masks
    _, m1 = cv2.threshold(p1, args.threshold, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(p2, args.threshold, 255, cv2.THRESH_BINARY)

    # 4) morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph, args.morph))
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, k)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, k)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k)

    # 5) compute differences
    appeared = cv2.bitwise_and(m2, cv2.bitwise_not(m1))
    disappeared = cv2.bitwise_and(m1, cv2.bitwise_not(m2))

    # 6) save masks
    out_appeared = args.out_prefix + "_appeared.png"
    out_disappeared = args.out_prefix + "_disappeared.png"
    out_mask1 = args.out_prefix + "_mask_before.png"
    out_mask2 = args.out_prefix + "_mask_after.png"
    cv2.imwrite(out_mask1, m1)
    cv2.imwrite(out_mask2, m2)
    cv2.imwrite(out_appeared, appeared)
    cv2.imwrite(out_disappeared, disappeared)
    print("Saved masks to:", out_mask1, out_mask2, out_appeared, out_disappeared)

    # 7) optional: create RGB overlay using image2's bands (if it's a 6-band TIFF)
    try:
        arr = tifffile.imread(args.image2)
        if arr.ndim == 3 and arr.shape[-1] == 6:
            rgba = to_uint8_rgb_from_6band(arr)
        elif arr.ndim == 3 and arr.shape[0] == 6:
            rgba = to_uint8_rgb_from_6band(np.moveaxis(arr, 0, -1))
        else:
            rgba = None
        if rgba is not None:
            # overlay appeared as red tint
            overlay = rgba.copy()
            mask_bool = appeared.astype(bool)
            overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) + 120, 0, 255)  # add red
            out_overlay = args.out_prefix + "_overlay_appeared.png"
            cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print("Saved overlay:", out_overlay)
    except Exception as e:
        print("Could not create overlay from image2 (need 6-band tif). Error:", e)

if __name__ == "__main__":
    main()