# Synthetic Data POC Report

## Overview
This document explains how the synthetic rack dataset was generated, which augmentation techniques were used, the selected model for the proof of concept, and the current dataset status.

## Synthetic data generation process
- Used `BlenderProc` together with `Blender 4.2.1 LTS`.
- The generation script is `scripts/rack_gen.py`.
- Device assets are loaded from `Assets/devices_front_only/`.
- Port placement and layout configuration are loaded from `config/port_locations.json`.
- Output images are saved under `outputs/` and copied into the POC dataset under `Dataset/images/`.

### Key generation details
- Each image is a synthetic rack view with rack devices placed in realistic rack slots.
- The renderer uses randomized rack fill, lighting, camera parameters, and texture variations.
- The dataset includes YOLO-format annotations generated from the 3D object positions.
- Labels are saved as `Dataset/labels/synthetic_rack_XXX.txt`.
- Class names are stored in `Dataset/classes.txt`.

## Augmentation techniques used
The dataset uses realistic image augmentations applied after rendering, while preserving YOLO bounding boxes.

1. **Exposure & contrast variation**
   - Random brightness adjustment
   - Random contrast scaling
2. **Color variation**
   - HSV saturation shift
   - HSV value shift
   - Color temperature adjustment via red/blue channel compensation
3. **Sensor / camera realism**
   - Random Gaussian noise
   - Mild Gaussian blur
4. **Lens effect**
   - Vignette darkening at image corners
5. **Geometric mirror augmentation**
   - Horizontal flip with label coordinate correction

These augmentations are designed to mimic real camera capture conditions while keeping the object annotations valid.

## Model selected for POC
- Model: `YOLOv8n` from Ultralytics
- Rationale:
  - Smallest YOLOv8 variant for fast proof-of-concept training
  - Good for quick validation during the meeting
  - Easy to upgrade later to `yolov8s` or `yolov8m` once the full dataset is ready

## Dataset status
- Current dataset folder: `Dataset/`
- Images: `Dataset/images/`
- Labels: `Dataset/labels/`
- YOLO config: `Dataset/data.yaml`
- Training script: `Dataset/train_poc.py`
- Evaluation script: `Dataset/evaluate_poc.py`

### Current dataset size
- Total images copied into `Dataset/images/`: 319
- Total label files copied into `Dataset/labels/`: 319

This dataset was created from the 84 base synthetic renders completed so far, plus their augmented variations.

## Training setup
The POC training script is `Dataset/train_poc.py`.

Example Colab command:
```bash
!pip install ultralytics
!python Dataset/train_poc.py --epochs 10 --batch 8 --imgsz 640
```

If you want a faster POC:
```bash
!python Dataset/train_poc.py --epochs 5 --batch 8 --imgsz 640
```

## Evaluation and results
The evaluation script is `Dataset/evaluate_poc.py`.
It can:
- report dataset statistics
- create train/val/test splits
- evaluate a trained YOLO model
- generate annotated prediction images
- write a markdown report in `Dataset/evaluate_results/`

Example evaluation command:
```bash
python Dataset/evaluate_poc.py --model runs/train_poc/rack_poc/weights/best.pt --output-dir Dataset/evaluate_results
```

### What this report shows
- total dataset counts
- annotation counts per class
- split sizes if split files exist
- model performance metrics when a trained model is provided
- sample predicted images saved under `Dataset/evaluate_results/predictions/`

## What is still pending
- Full-generation dataset target: 1000 base images
- Stable retraining once the full dataset is complete
- Final model performance metrics after the complete dataset is trained

## Meeting POC recommendation
For the meeting, use the current `Dataset/` folder and train `YOLOv8n` as a fast POC.

The generated report and evaluation scripts will provide:
- dataset summary
- annotations per class
- split information
- model validation metrics
- sample detection output images

Once the full generation completes, rerun `Dataset/train_poc.py` and `Dataset/evaluate_poc.py` on the completed dataset for the final stable model.
