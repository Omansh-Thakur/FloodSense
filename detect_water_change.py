import os, sys, argparse
import numpy as np
import tifffile as tiff
import cv2
import math
from collections import deque
from scipy import ndimage as ndi

#import rasterio for aligning image
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

import floodSense
import glob
import glob as _glob


# loading landset bands
def load_landsat_bands(folder_path):
    band_order = ["B2", "B3", "B4", "B5", "B6", "B7"]
    band_arrays = []

    for band in band_order:
        patterns = [
            os.path.join(folder_path, f"*_{band}.tif"),
            os.path.join(folder_path, f"*_{band}.TIF"),
            os.path.join(folder_path, f"*{band}.tif"),
            os.path.join(folder_path, f"*{band}.TIF"),
        ]
        matches = []
        for p in patterns:
            matches.extend(glob.glob(p))

        if not matches:
            raise FileNotFoundError(f"Missing band {band} in folder: {folder_path}")

        band_arrays.append(tiff.imread(matches[0]).astype(np.float32))

    shapes = {a.shape for a in band_arrays}
    if len(shapes) != 1:
        raise ValueError(f"band shapes differ: {shapes}")

    return np.stack(band_arrays, axis=-1)


# Reading Input

def read_input(path):
    if os.path.isdir(path):
        arr = load_landsat_bands(path)
        return arr, None
    else:
        if HAS_RASTERIO:
            src = rasterio.open(path)
            arr = np.transpose(src.read([1,2,3,4,5,6]), (1,2,0)).astype(np.float32)
            profile = src.profile
            src.close()
            return arr, profile
        else:
            arr = tiff.imread(path).astype(np.float32)
            return arr, None


# Normalizing reflectance (DN → SR)
def normalize_reflectance(img, force_dn=False):
    img = img.astype(np.float32)
    img = np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    raw_max = float(np.max(img))
    if force_dn or raw_max > 2.0:
        img = img * 0.0000275 - 0.2

    img = np.clip(img, 0.0, 1.0)
    return img


def tiled_predict(model, img, tile=512, overlap=64, batch_size=1, force_dn=False):
    H, W, C = img.shape
    stride = tile - overlap

    n_rows = int(math.ceil((H - overlap) / stride))
    n_cols = int(math.ceil((W - overlap) / stride))
    pad_h = max(0, n_rows * stride + overlap - H)
    pad_w = max(0, n_cols * stride + overlap - W)

    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    out = np.zeros((img_pad.shape[0], img_pad.shape[1]), dtype=np.float32)
    weight = np.zeros_like(out)

    tiles = []
    coords = []

    for r in range(0, img_pad.shape[0] - overlap, stride):
        for c in range(0, img_pad.shape[1] - overlap, stride):
            patch = img_pad[r:r+tile, c:c+tile, :]
            patch = normalize_reflectance(patch, force_dn=force_dn)
            tiles.append(patch)
            coords.append((r, c))

            # batch run
            if len(tiles) >= batch_size:
                batch = np.stack(tiles, axis=0)
                preds = model.predict(batch)
                preds = np.squeeze(preds)

                for i, (rr, cc) in enumerate(coords):
                    p = preds[i]
                    out[rr:rr+tile, cc:cc+tile] += p
                    weight[rr:rr+tile, cc:cc+tile] += 1.0

                tiles = []
                coords = []

    if len(tiles) > 0:
        batch = np.stack(tiles, axis=0)
        preds = model.predict(batch)
        preds = np.squeeze(preds)

        for i, (rr, cc) in enumerate(coords):
            p = preds[i]
            out[rr:rr+tile, cc:cc+tile] += p
            weight[rr:rr+tile, cc:cc+tile] += 1.0

    out = out / np.maximum(weight, 1e-6)
    out = out[:H, :W]
    return out


# Remove small areas
def postprocess_mask(mask_uint8, min_area_px=256):
    labeled, n = ndi.label(mask_uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    remove = sizes < min_area_px
    if remove.sum() > 0:
        mask_clean = mask_uint8.copy()
        for lab_i, rem in enumerate(remove):
            if rem:
                mask_clean[labeled == lab_i] = 0
        return mask_clean

    return mask_uint8


# Save geotiff
def save_geotiff(path, arr, profile):
    if not HAS_RASTERIO or profile is None:
        tiff.imwrite(path, arr.astype(np.float32))
    else:
        outp = profile.copy()
        outp.update(dtype='float32', count=1, compress='lzw')
        with rasterio.open(path, 'w', **outp) as dst:
            if arr.ndim == 2:
                dst.write(arr.astype(np.float32), 1)
            else:
                for i in range(arr.shape[2]):
                    dst.write(arr[:,:,i].astype(np.float32), i+1)


def main(args):

    # load model
    model = floodSense.model()
    model.load_weights(args.checkpoint)

    imgA, profA = read_input(args.sceneA)
    imgB, profB = read_input(args.sceneB)

    # check alignment
    if HAS_RASTERIO and profA and profB:
        if profA.get('transform') != profB.get('transform') \
           or profA.get('crs') != profB.get('crs') \
           or profA['height'] != profB['height'] \
           or profA['width'] != profB['width']:
            print("[WARN] Geo metadata differ. You should reproject/resample.")
    else:
        if imgA.shape != imgB.shape:
            raise ValueError("Input images have different shapes. Align first.")

    # inference
    print("[INFO] Running inference on scene A")
    probA = tiled_predict(model, imgA, tile=args.tile, overlap=args.overlap,
                          batch_size=args.batch, force_dn=args.force_dn)

    print("[INFO] Running inference on scene B")
    probB = tiled_predict(model, imgB, tile=args.tile, overlap=args.overlap,
                          batch_size=args.batch, force_dn=args.force_dn)

    # save probability maps
    os.makedirs(args.out_dir, exist_ok=True)

    probA_path = os.path.join(args.out_dir, "prob_A.tif")
    probB_path = os.path.join(args.out_dir, "prob_B.tif")
    save_geotiff(probA_path, probA, profA)
    save_geotiff(probB_path, probB, profB)

    print(f"[SAVE] prob maps saved: {probA_path}, {probB_path}")

    # finding difference 
    diff = probB - probA
    diff_path = os.path.join(args.out_dir, "diff.tif")
    save_geotiff(diff_path, diff, profA)
    print(f"[SAVE] diff saved: {diff_path}")

    # masks
    thr = args.thr
    new_water = ((probB >= thr) & (probA < thr)).astype(np.uint8)
    lost_water = ((probA >= thr) & (probB < thr)).astype(np.uint8)
    persistent = ((probA >= thr) & (probB >= thr)).astype(np.uint8)

    new_water = postprocess_mask(new_water, min_area_px=args.min_area_px)
    lost_water = postprocess_mask(lost_water, min_area_px=args.min_area_px)
    persistent = postprocess_mask(persistent, min_area_px=args.min_area_px)

    for name, mask in [
        ('new_water', new_water),
        ('lost_water', lost_water),
        ('persistent_water', persistent)
    ]:
        out_mask_path = os.path.join(args.out_dir, f"{name}.tif")
        save_geotiff(out_mask_path, mask.astype(np.uint8), profA)
        vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_vis.png"), vis)
        print(f"[SAVE] {name} -> {out_mask_path}")

    print("[DONE] Change detection complete. Outputs in:", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sceneA", required=True, help="folder or 6-band tif for earlier date")
    parser.add_argument("--sceneB", required=True, help="folder or 6-band tif for later date")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--min_area_px", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--force_dn", action='store_true', help="force DN->reflectance scaling")

    args = parser.parse_args()
    main(args)
