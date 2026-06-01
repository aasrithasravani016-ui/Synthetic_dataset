# Synthetic Rack Dataset Generation Guide

## Overview
This project generates **synthetic training data for object detection** using Blender. It creates realistic images of server racks with labeled components for training a machine learning model.

---

## What Gets Generated

### Files Created
After running the generation script, you will have:

```
outputs/
├── classes.txt                          # List of 4 object classes
├── synthetic_rack_000.png               # Original rendered image 1
├── synthetic_rack_000_aug0.png          # Augmented version 1 of image 1
├── synthetic_rack_000_aug1.png          # Augmented version 2 of image 1
├── synthetic_rack_000_aug2.png          # Augmented version 3 of image 1
├── ...
└── labels/
    ├── synthetic_rack_000.txt           # Labels for image 1
    ├── synthetic_rack_000_aug0.txt      # Labels for augmented version 1
    ├── synthetic_rack_000_aug1.txt      # Labels for augmented version 2
    ├── synthetic_rack_000_aug2.txt      # Labels for augmented version 3
    └── ...
```

**Total Generated:**
- 1,000 base images rendered from Blender
- 2-4 augmented versions per image
- **~3,000 total images with labels**

---

## Object Classes

The model will learn to detect **4 classes of devices**:

| Class ID | Name | What It Is |
|----------|------|-----------|
| 0 | `patch_panel` | Network connection panel in rack |
| 1 | `pdu` | Power distribution unit |
| 2 | `switch` | Network switch |
| 3 | `server` | Server unit |

---

## Label Format (YOLO Format)

Each `.txt` label file contains bounding boxes in this format:

```
<class_id> <center_x> <center_y> <width> <height>
```

**Example:**
```
2 0.45 0.32 0.20 0.15
0 0.65 0.50 0.18 0.10
```

Where:
- `2` = class ID (switch in this case)
- `0.45` = center X position (normalized 0-1)
- `0.32` = center Y position (normalized 0-1)
- `0.20` = box width (normalized 0-1)
- `0.15` = box height (normalized 0-1)

All values are **normalized to 0-1 range** relative to image dimensions.

---

## Running the Generation Script

### Prerequisites
- Blender 4.2.1 or later
- BlenderProc library installed
- Python 3.11+
- OpenCV (cv2)
- NumPy

### Command to Run

```bash
env BLENDERPROC_CUSTOM_PACKAGES_PATH=/Users/aasritha/Downloads/Synthetic/blenderproc_packages \
    BLENDERPROC_SKIP_EMBEDDED_PIP_SETUP=1 \
    blenderproc run /Users/aasritha/Downloads/Synthetic/rack_gen.py
```

### What Happens
1. Loads device textures (switches, patch panels, PDUs, servers)
2. For each of 1,000 iterations:
   - Randomly generates a realistic rack layout
   - Renders high-quality image (1024×1400 pixels)
   - Creates 3D to 2D projections for bounding boxes
   - Saves original image and labels
   - Generates 2-4 augmented versions automatically
3. Prints progress: `=== Generating image X/1000 ===`

**Estimated Time:** 4-8 hours depending on your GPU

---

## Augmentation Techniques Applied

Each augmented image is randomly modified with:

| Technique | Details |
|-----------|---------|
| **Brightness/Contrast** | ±30 brightness, 0.8-1.3x contrast (60% chance) |
| **Color Saturation** | 0.7-1.3x saturation, 0.85-1.15x value (50% chance) |
| **Blur** | Random Gaussian blur 3×3 or 5×5 (30% chance) |
| **Horizontal Flip** | Mirror image + flip bboxes (50% chance) |
| **Perspective Transform** | Slight 3D perspective warp (30% chance) |
| **Rotation** | ±3 degrees random rotation (40% chance) |

**Benefit:** Trains your model to be robust to different lighting, angles, and image conditions.

---

## Using the Dataset with Google Colab

### Step 1: Upload to Google Drive
1. Zip the `outputs/` folder
2. Upload to Google Drive
3. Mount Drive in Colab

### Step 2: Create YOLO Dataset Config

Create `data.yaml`:

```yaml
path: /path/to/outputs
train: images  # We'll split the images 80-20
val: images
test: images

nc: 4
names: ['patch_panel', 'pdu', 'switch', 'server']
```

### Step 3: Install YOLOv8 in Colab

```python
!pip install ultralytics opencv-python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')  # nano model (fast, small)
# or use 'yolov8s.pt', 'yolov8m.pt' for larger models

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    device=0,  # GPU device
    patience=20  # Early stopping
)

# Evaluate
metrics = model.val()

# Make predictions
results = model.predict(source='test_image.jpg')
```

### Step 4: Export Model

```python
# Export to different formats
model.export(format='onnx')   # For inference
model.export(format='pt')     # PyTorch
model.export(format='tflite') # Mobile
```

---

## Rack Dataset Details

### What Each Rack Contains

Each randomly generated rack has:
- **35-42 occupied slots** (83%+ fill rate, like real data centers)
- **8-14 switches** (bulk of the network connectivity)
- **2-3 patch panels** (aggregation points)
- **2 servers** (2U each)
- **2 PDUs** (power distribution)
- **Cables** connecting adjacent devices
- **LEDs** on some devices (realistic details)
- **Cable management bars** (real-world details)

### Realistic Details
- High-quality 3D models with proper materials
- Proper lighting simulation (fluorescent ceiling, fill lights)
- Network cables with color variation
- RJ45 connectors
- Proper depth of field (camera DoF)
- Camera noise and vignetting
- Color temperature shifts

---

## File Structure Explanation

### Input Files Needed
```
Assets/
└── devices_front_only/
    ├── switches/          # PNG images of switch fronts
    ├── patch_panel/       # PNG images of patch panels
    ├── pdu/               # PNG images of PDUs
    └── Server/            # PNG images of servers

port_locations.json       # Pre-calculated port positions on devices
```

### Output Files Generated
```
outputs/
├── classes.txt                    # Class list for training
├── synthetic_rack_*.png           # Raw rendered images
├── synthetic_rack_*_aug*.png      # Augmented versions
└── labels/
    ├── synthetic_rack_*.txt       # YOLO format bboxes
    └── synthetic_rack_*_aug*.txt  # Augmented labels
```

---

## Training Tips for Colab

### Recommended Settings
```python
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,           # Resize to 640×640 (standard)
    batch=16,            # Batch size (adjust for GPU memory)
    patience=20,         # Stop if no improvement for 20 epochs
    device=0,            # Use first GPU
    amp=True,            # Mixed precision (faster)
    mosaic=True,         # YOLOv8 data augmentation
)
```

### Model Sizes
- **YOLOv8n** (nano): Fast, small, for real-time detection
- **YOLOv8s** (small): Balanced speed/accuracy
- **YOLOv8m** (medium): Better accuracy, slower
- **YOLOv8l** (large): Best accuracy, requires more GPU memory

### Expected Performance
With ~3,000 synthetic images:
- mAP50 should reach 60-75% after training
- Real-world performance depends on actual rack images

---

## Troubleshooting

### Script Stops Early
**Cause:** Memory error or Blender crash
**Solution:** Restart Blender, reduce image resolution in script

### Wrong Image Dimensions
**Cause:** Mismatch between render resolution and projection
**Solution:** Script automatically handles this, just verify OUTPUT_DIR exists

### Labels Look Wrong
**Cause:** Camera matrix calculation error
**Solution:** Ensure Blender camera is set up correctly (script does this)

### Colab Training is Slow
**Cause:** Using CPU or small GPU
**Solution:** 
- Ensure GPU is enabled: `!nvidia-smi`
- Use smaller model (yolov8n)
- Reduce batch size

---

## Performance Expectations

### Generation Time
- **Per image:** ~20-30 seconds (rendering is the bottleneck)
- **1,000 base images:** 6-8 hours
- **With augmentation:** Total time same (augmentation is instant post-processing)

### Generated Dataset Size
- **Raw images:** ~2GB for 3,000 images
- **With labels:** ~2GB (labels are tiny text files)
- **Total:** ~4GB including classes.txt

### Training Time (Colab GPU)
- **YOLOv8n:** 30-45 minutes for 100 epochs
- **YOLOv8s:** 1-2 hours for 100 epochs
- **YOLOv8m:** 2-4 hours for 100 epochs

---

## Next Steps

1. **Run generation script** on your machine
2. **Verify outputs folder** has ~3,000 images and labels
3. **Zip and upload** to Google Drive
4. **Create Colab notebook** with training code above
5. **Train model** (will take 30 mins to 4 hours depending on model)
6. **Deploy** to your application

---

## Common Questions

**Q: Can I use other Blender versions?**
A: Yes, but texturing and materials may look different. 4.2+ recommended.

**Q: Can I modify the rack layout?**
A: Yes, edit `rack_gen.py` - look for `num_switches`, `num_patch`, `num_servers`, `num_pdus` values.

**Q: Can I add more classes?**
A: Yes, add to `CLASSES` dictionary and add device types to the Assets folder.

**Q: How do I use the trained model?**
A: See YOLOv8 docs. Basic inference: `results = model.predict('image.jpg')`

**Q: Can I export the model to mobile?**
A: Yes, use `model.export(format='tflite')` for TensorFlow Lite (mobile).

---

## Support

For issues with:
- **BlenderProc:** Check blenderproc.org documentation
- **YOLOv8:** Check github.com/ultralytics/ultralytics
- **This script:** Review comments in `rack_gen.py`

---

**Generated:** June 2024  
**Dataset Version:** 1.0 (Synthetic Rack Detection)
