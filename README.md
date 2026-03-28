CNN-based flood and surface water detection from multispectral satellite imagery.

# FloodSense

FloodSense is a machine learning project for **surface water segmentation** and **flood change detection** from multispectral satellite images. It leverages a pretrained CNN model (DeepWaterMap【19†L364-L365】) to identify water pixels and then compares two scenes to find **new**, **lost**, and **persistent** water areas. The pipeline covers reading multispectral data, tile-based inference, thresholding, and generating output masks and visualisations.

## Project Highlights

- **Pretrained CNN model (DeepWaterMap)** for water segmentation【19†L364-L365】.  
- Processes **6-band Landsat imagery** (Blue, Green, Red, NIR, SWIR1, SWIR2)【9†L306-L311】【13†L312-L317】.  
- **Tile-based inference** allows very large images (multi-megapixel) to be processed.  
- **Change detection** between two dates: identifies new water, lost water, and persistent water.  
- Basic **post-processing** (thresholding, connected-component filtering) cleans the raw masks.  
- Outputs **GeoTIFF masks** and **visualisation overlays**.

## Example Results

*(Add example images here: before/after satellite scenes, water masks, change overlay)*

- **Before (Scene A):** Placeholder for input satellite image before flood.  
- **After (Scene B):** Placeholder for input satellite image after flood.  
- **Water Mask (Scene A):** Placeholder for predicted water mask (scene A).  
- **Water Mask (Scene B):** Placeholder for predicted water mask (scene B).  
- **Change Overlay:** Placeholder for overlay showing new water (white) and lost water (black).

## Model Architecture

FloodSense uses an **encoder–decoder CNN (U-Net style)** for segmentation. It takes a 6-channel input (the spectral bands) and outputs a pixel-wise water probability map. The basic flow is:

```
Multispectral Image (6 bands: B2..B7)
        ↓
    CNN Encoder
   (feature extraction)
        ↓
    Bottleneck
   (context features)
        ↓
    CNN Decoder
 (spatial reconstruction)
        ↓
Water Probability Map (floating-point per pixel)
        ↓
Binary Water Mask (thresholded)
```

The model was defined in TensorFlow and expects input bands in the Landsat order. According to the original DeepWaterMap code: **B2: Blue, B3: Green, B4: Red, B5: NIR, B6: SWIR1, B7: SWIR2**【13†L312-L317】.

## Spectral Bands

| Band | Description                  |
|------|------------------------------|
| B2   | Blue                         |
| B3   | Green                        |
| B4   | Red                          |
| B5   | Near Infrared (NIR)          |
| B6   | Shortwave Infrared 1 (SWIR1) |
| B7   | Shortwave Infrared 2 (SWIR2) |

*(These correspond to Landsat-8 OLI band designations【9†L306-L311】【13†L312-L317】.)*

## Dataset Format

FloodSense accepts multispectral data in two formats:

1. **Single 6-band GeoTIFF**: One file (e.g. `scene.tif`) with six stacked bands (B2–B7).  
2. **Folder of individual band files**: A directory containing six separate GeoTIFFs named (e.g. `image_B2.tif` through `image_B7.tif`). The code will read and stack them in the correct order【13†L327-L330】.

Example folder structure for scene A:

```
sceneA/
    image_B2.tif
    image_B3.tif
    image_B4.tif
    image_B5.tif
    image_B6.tif
    image_B7.tif
```

The same applies for `sceneB/` (the later-date scene).

## Sample Data

- **sample_data/A/** – Earlier scene (e.g. before flood).  
- **sample_data/B/** – Later scene (e.g. after flood).  
- **sample_data/B_aligned/** – Optional: Scene B reprojected/aligned to match Scene A’s grid.  
- **sample_data/output/** – Example outputs (GeoTIFF masks and PNG visualisations).

These illustrate the pipeline. If the two scenes have different CRS or resolution, align them (e.g. using GDAL) so that the pixels line up.

## Installation

```bash
git clone https://github.com/Omansh-Thakur/FloodSense.git
cd FloodSense
pip install -r requirements.txt
```
*(Requires Python 3 and the libraries listed in `requirements.txt`.)*

## Running

Run the change detection script with your two scenes and the model checkpoint:

```bash
python src/detect_water_change.py \
  --checkpoint models/floodsense_checkpoint.ckpt \
  --sceneA sample_data/A \
  --sceneB sample_data/B \
  --out_dir sample_data/output
```

This uses default parameters. You can add options like `--thr 0.5` (water threshold), `--tile 512`, `--batch 4`, etc. For example:

```bash
python src/detect_water_change.py \
  --checkpoint models/floodsense_checkpoint.ckpt \
  --sceneA sample_data/A \
  --sceneB sample_data/B \
  --out_dir sample_data/output \
  --thr 0.5 --tile 512 --batch 4
```

The results will be saved in `sample_data/output/`.

## Outputs

After running, the output directory will contain:

| File                    | Description                            |
|-------------------------|----------------------------------------|
| `prob_A.tif`            | Water probability map (scene A)        |
| `prob_B.tif`            | Water probability map (scene B)        |
| `diff.tif`              | Difference (prob_B - prob_A)           |
| `new_water.tif`         | New water mask (1 = water in B only)   |
| `lost_water.tif`        | Lost water mask (1 = water in A only)  |
| `persistent_water.tif`  | Persistent water mask (1 = water in both) |
| `new_water_vis.png`     | New water (white) visualisation        |
| `lost_water_vis.png`    | Lost water (white) visualisation       |
| `persistent_water_vis.png` | Persistent water (white) visualisation |

These show the detected water areas and changes between the scenes.

## Repository Structure

```
FloodSense/
│
├── floodSense.py
├── detect_water_change.py
├── diff_inference.py
│
├── Screenshots/
│
├── diagrams/           (architecture/pipeline diagrams)
│   └── architecture.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Checkpoints

The pretrained model checkpoint is **not included** in this repository (it is large).
```
The pretrained model checkpoint (from DeepWaterMap) is not included.
Download it from: https://utexas.app.box.com/s/j9ymvdkaq36tk04be680mbmlaju08zkq/file/565662752887
Place the file as: models/floodsense_checkpoint.ckpt
```

## Troubleshooting

- **Scene alignment**: If your scenes have different CRS or resolution, align scene B to scene A (e.g. using `gdalwarp`) so that pixels correspond.  
- **Memory errors**: Reduce tile size (`--tile`) or batch size (`--batch`), or run on a machine with more RAM/GPU memory. You can also run on CPU if needed.  
- **Incorrect outputs**: Verify the input band order (B2..B7) and that `--sceneA` truly points to the earlier date.  
- **Dependencies**: Ensure all required libraries (TensorFlow, Rasterio, etc.) are installed in your Python environment.

## Future Improvements

- Integrate Sentinel-1 SAR data for all-weather flood detection.  
- Experiment with newer segmentation architectures (e.g. Vision Transformers).  
- Automate cloud masking or more robust post-processing.  
- Develop a GUI or web interface for real-time flood monitoring.

## Contributions

- Adapted a pretrained DeepWaterMap segmentation model【19†L364-L365】 to build an end-to-end flood detection pipeline using Landsat imagery.  
- Implemented tiled CNN inference and temporal change detection between scenes (new/lost/persistent water).  
- Developed geospatial data handling and visualisation (GeoTIFF outputs, overlay images).  
- Demonstrated integration of ML and remote sensing in a project setting.

## Acknowledgements

This project uses the DeepWaterMap model (Isikdogan et al.)【19†L364-L365】 and its pretrained checkpoint (trained for 135 epochs on Landsat data【19†L358-L360】). DeepWaterMap is available on GitHub [(deepwatermap)](https://github.com/isikdogan/deepwatermap.git)【19†L364-L365】. The code here focuses on building the data pipeline, change detection, and visualisation around that model.

## Author

Omansh – B.Tech Computer Science.  

**Sources:** Landsat band designations (USGS)【9†L306-L311】; DeepWaterMap GitHub repository【13†L312-L317】【19†L358-L365】.
