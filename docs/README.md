# Synthetic Rack Dataset for Object Detection

Generate realistic synthetic training data for detecting server rack components with automatic augmentation.

---

## 🎯 Quick Start

### 1. Clean Up Workspace
```bash
# Remove unnecessary files (see ORGANIZATION_PLAN.md for details)
# This frees ~15-20 GB by removing duplicate Blender installations
```

### 2. Generate Dataset (6-8 hours)
```bash
cd /Users/aasritha/Downloads/Synthetic

env BLENDERPROC_CUSTOM_PACKAGES_PATH=/Users/aasritha/Downloads/Synthetic/blenderproc_packages \
    BLENDERPROC_SKIP_EMBEDDED_PIP_SETUP=1 \
    blenderproc run rack_gen.py
```

### 3. Upload to Google Drive
```bash
# Zip the outputs folder
cd outputs
zip -r synthetic_dataset.zip *

# Upload to Google Drive
```

### 4. Train in Google Colab
See `COLAB_TRAINING_GUIDE.md` for complete Colab notebook code

---

## 📊 What You Get

**Generated Dataset:**
- 1,000 synthetic images
- 2-4 augmented versions per image
- **~3,000 total images** with YOLO format labels
- 4 classes: patch_panel, pdu, switch, server
- 1024×1400 pixel high-quality renders

**Label Format (YOLO):**
```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates normalized to 0-1.

---

## 📚 Documentation

### For Setup & Cleanup
→ **`ORGANIZATION_PLAN.md`** - Delete unnecessary files, organize directory structure

### For Dataset Generation
→ **`DATASET_GENERATION_GUIDE.md`** - Complete guide with:
- What gets generated
- How augmentation works
- Rack composition details
- Troubleshooting

### For Colab Training
→ **`COLAB_TRAINING_GUIDE.md`** - Copy-paste Colab notebook with:
- Step-by-step cells
- Training setup
- Model validation
- Inference examples

---

## 🏃 Running the Generator

### System Requirements
- Blender 4.2+
- 8+ GB RAM
- GPU recommended (CUDA/Metal)
- Python 3.11+

### One Command
```bash
env BLENDERPROC_CUSTOM_PACKAGES_PATH=/Users/aasritha/Downloads/Synthetic/blenderproc_packages \
    BLENDERPROC_SKIP_EMBEDDED_PIP_SETUP=1 \
    blenderproc run rack_gen.py
```

### What Happens
1. Loads device textures (switches, PDUs, patch panels, servers)
2. For each of 1,000 iterations:
   - Generates random rack layout
   - Renders realistic image
   - Creates bounding box labels
   - Generates 2-4 augmented versions
3. Saves to `outputs/` with `labels/` subfolder

### Expected Output
```
outputs/
├── classes.txt                 # Class definitions
├── synthetic_rack_000.png      # Image 1
├── synthetic_rack_000_aug0.png # Augmented version 1
├── synthetic_rack_000_aug1.png # Augmented version 2
├── synthetic_rack_000_aug2.png # Augmented version 3
├── ...
└── labels/
    ├── synthetic_rack_000.txt
    ├── synthetic_rack_000_aug0.txt
    ├── synthetic_rack_000_aug1.txt
    └── ...
```

---

## 🖥️ Using with Google Colab

### Steps
1. Generate dataset locally (you're here)
2. Zip `outputs/` folder
3. Upload to Google Drive
4. Open `COLAB_TRAINING_GUIDE.md`
5. Copy cells into Colab notebook
6. Train YOLOv8 model

### Training in Colab
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='data.yaml', epochs=100)
```

Expected training time: **30 mins - 4 hours** (depends on model size)

---

## 🎨 Augmentation Applied

Each image is randomly modified with:
- Brightness/contrast shifts
- Color saturation changes
- Gaussian blur
- Horizontal flips
- Perspective transforms
- Rotation (±3°)

This creates ~3,000 diverse training images from 1,000 renders.

---

## 📋 Object Classes

Your model will detect:

| ID | Class | Example |
|----|-------|---------|
| 0 | patch_panel | Network aggregation point |
| 1 | pdu | Power distribution unit |
| 2 | switch | Network switch |
| 3 | server | Server unit |

---

## ⚠️ Important Files

**DO NOT DELETE:**
- `rack_gen.py` - Main generation script
- `Assets/devices_front_only/` - Device textures
- `port_locations.json` - Port position data
- `blenderproc_packages/` - Dependencies

**OK to DELETE:**
- Extra Blender installations
- Old output directories (v1, v2, outputs_1, etc.)
- Blender .dmg file

See `ORGANIZATION_PLAN.md` for detailed cleanup.

---

## 🔧 Customization

Edit these in `rack_gen.py`:

```python
# Change number of base images to generate
NUM_IMAGES = 1000  # Default

# Adjust rack density
fill_target = int(TOTAL_SLOTS * 0.85)  # 85% full

# Modify device distribution
num_switches = random.randint(8, 14)   # Range: 8-14 switches
num_patch = random.randint(2, 3)       # Range: 2-3 patch panels
num_servers = 2                        # Fixed: 2 servers
num_pdus = 2                           # Fixed: 2 PDUs

# Adjust augmentation count
num_augmentations = random.randint(2, 4)  # Range: 2-4 per image
```

---

## 📈 Performance Expectations

| Task | Time | Hardware |
|------|------|----------|
| Generate 1,000 images | 6-8 hours | GPU recommended |
| Train YOLOv8n | 30-45 min | Colab GPU |
| Train YOLOv8s | 1-2 hours | Colab GPU |
| Train YOLOv8m | 2-4 hours | Colab GPU |

**Dataset Size:** ~2 GB images + ~100 MB labels = ~2.5 GB total

---

## 🐛 Troubleshooting

**Generation stops/errors:**
- Check Blender path is correct
- Verify `Assets/devices_front_only/` has images
- Check disk space (need ~5 GB for outputs)

**Colab training slow:**
- Use `yolov8n.pt` (nano model)
- Reduce batch size to 8
- Ensure GPU is enabled: `!nvidia-smi`

**Labels look wrong:**
- Camera projection uses 3D→2D conversion
- Script automatically handles this
- If issues persist, check Blender render resolution

**Out of memory during training:**
```python
# In Colab training
batch=8  # Reduce from 16
imgsz=512  # Reduce from 640
```

---

## 📞 Support

### Error in dataset generation
→ See `DATASET_GENERATION_GUIDE.md` troubleshooting section

### Error in Colab training
→ See `COLAB_TRAINING_GUIDE.md` troubleshooting section

### Want to modify script
→ Comments in `rack_gen.py` explain all major sections

### Need different classes
→ Add to `CLASSES` dict and add device images to `Assets/devices_front_only/`

---

## 📝 File Reference

| File | Purpose |
|------|---------|
| `rack_gen.py` | Main dataset generation script |
| `DATASET_GENERATION_GUIDE.md` | Complete documentation |
| `COLAB_TRAINING_GUIDE.md` | Google Colab training code |
| `ORGANIZATION_PLAN.md` | Cleanup and directory organization |
| `README.md` | This file |
| `SYNTHETIC_DATA_REPORT.md` | Original project report |
| `Assets/` | Device textures |
| `outputs/` | Generated dataset (created after running) |

---

## 🚀 Next Steps

1. **Read** `ORGANIZATION_PLAN.md` and clean up workspace
2. **Run** the generation script
3. **Upload** results to Google Drive
4. **Follow** `COLAB_TRAINING_GUIDE.md` in Colab
5. **Deploy** your trained model!

---

**Status:** Ready to generate dataset and train detection model

**Generated:** June 2024  
**Version:** 1.0
