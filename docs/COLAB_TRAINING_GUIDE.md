# Colab Training Template

This is a complete Colab notebook code for training a YOLOv8 object detection model on the synthetic rack dataset.

---

## Copy-Paste Into Colab Cells

### Cell 1: Install Requirements
```python
!pip install ultralytics opencv-python pyyaml -q
```

### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 3: Extract Dataset
```python
import os
import zipfile

# Extract the zipped dataset
dataset_zip = '/content/drive/My Drive/synthetic_dataset.zip'  # Adjust path
extract_path = '/content/dataset'

if os.path.exists(dataset_zip):
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✓ Extracted to {extract_path}")
    
    # Check what we have
    for root, dirs, files in os.walk(extract_path):
        level = root.replace(extract_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files)-5} more')
else:
    print(f"✗ File not found: {dataset_zip}")
```

### Cell 4: Create YOLO Dataset Config
```python
# Read the classes from the dataset
classes_file = '/content/dataset/outputs/classes.txt'
with open(classes_file, 'r') as f:
    classes = f.read().strip().split('\n')
    
num_classes = len(classes)
print(f"✓ Found {num_classes} classes: {classes}")

# Create data.yaml for YOLO
data_yaml = f"""path: /content/dataset/outputs
train: images
val: images

nc: {num_classes}
names: {classes}
"""

with open('/content/data.yaml', 'w') as f:
    f.write(data_yaml)
    
print("✓ Created data.yaml")
print(data_yaml)
```

### Cell 5: Prepare Images and Labels
```python
import shutil
from pathlib import Path

# Create expected directory structure
output_dir = Path('/content/dataset/outputs')
images_dir = output_dir / 'images'
labels_dir = output_dir / 'labels'

# Make sure directories exist
images_dir.mkdir(exist_ok=True)

# Move PNG files to images directory
png_files = list(output_dir.glob('*.png'))
for png_file in png_files:
    if not (images_dir / png_file.name).exists():
        shutil.move(str(png_file), str(images_dir / png_file.name))

print(f"✓ Organized {len(list(images_dir.glob('*.png')))} images")
print(f"✓ Found {len(list(labels_dir.glob('*.txt')))} label files")
```

### Cell 6: Load Model and Train
```python
from ultralytics import YOLO
import torch

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")

# Load pretrained model
model = YOLO('yolov8n.pt')  # nano model - fast training
# Alternative: use 'yolov8s.pt' for better accuracy

# Train the model
results = model.train(
    data='/content/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,              # Reduce to 8 if out of memory
    device=0,              # GPU device
    patience=20,           # Early stopping
    amp=True,              # Faster training
    mosaic=True,           # YOLOv8 augmentation
    flipud=0.5,            # Vertical flip 50%
    fliplr=0.5,            # Horizontal flip 50%
    hsv_h=0.015,           # HSV-Hue augmentation
    hsv_s=0.7,             # HSV-Saturation
    hsv_v=0.4,             # HSV-Value
    degrees=10,            # Rotation degrees
    translate=0.1,         # Translation
    scale=0.5,             # Scaling
    perspective=0.0,       # Perspective
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    name='rack_detector'
)

print("✓ Training complete!")
```

### Cell 7: Validate Model
```python
# Validate on the entire dataset
metrics = model.val()

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

### Cell 8: Test on Sample Images
```python
from IPython.display import Image
import cv2

# Get a test image
test_images = list(images_dir.glob('*.png'))[:5]

for img_path in test_images:
    # Run inference
    results = model.predict(str(img_path), conf=0.4)
    
    # Get results
    result = results[0]
    
    # Plot results
    result.show()
    
    # Print detections
    print(f"\n{img_path.name}:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = classes[class_id]
        print(f"  - {class_name}: {confidence:.2%}")
```

### Cell 9: Export Model
```python
# Export to different formats
print("Exporting model...")

# PyTorch (smallest, fastest)
model.export(format='pt')

# ONNX (universal format)
model.export(format='onnx')

# TensorFlow Lite (mobile)
model.export(format='tflite')

# CoreML (iOS)
model.export(format='coreml')

print("✓ Exported to:")
print("  - .pt (PyTorch)")
print("  - .onnx (ONNX)")
print("  - .tflite (Mobile)")
print("  - .mlmodel (iOS)")

# Download from Google Drive > /content/runs/detect/rack_detector/weights/
```

### Cell 10: Make Predictions on New Images
```python
# Download and predict on a new image
from IPython.display import Image as IPImage
import requests

# Example: predict on your own image
# Option 1: Use image from URL
url = "https://example.com/your_rack_image.jpg"
img_path = "/content/test_image.jpg"
# urllib.request.urlretrieve(url, img_path)

# Option 2: Upload your own
# from google.colab import files
# uploaded = files.upload()
# img_path = list(uploaded.keys())[0]

# Predict
results = model.predict(img_path, conf=0.5)
result = results[0]

# Show results
result.show()

# Print detections
print("Detections:")
for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    print(f"  {classes[class_id]}: {confidence:.1%}")
```

### Cell 11: Download Results
```python
import shutil

# Create a zip of results
results_path = '/content/runs/detect/rack_detector'
output_zip = '/content/rack_detector_results.zip'

shutil.make_archive(
    output_zip.replace('.zip', ''),
    'zip',
    results_path
)

# Download
from google.colab import files
files.download(output_zip)

print("✓ Downloaded results.zip")
```

---

## Quick Start Steps

1. **Upload Dataset**
   - Generate on local machine using `rack_gen.py`
   - Zip the `outputs/` folder
   - Upload to Google Drive

2. **Create Colab Notebook**
   - New -> Python notebook
   - Copy cells above in order

3. **Run Cells**
   - Cell 1-5: Setup (~2 min)
   - Cell 6: Training (~30-120 min depending on model)
   - Cell 7-10: Validation and testing

4. **Download Model**
   - Use Cell 11 to download weights
   - Use in your own application

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size in Cell 6
batch=8  # instead of 16
```

### Slow Training
```python
# Use smaller model
model = YOLO('yolov8n.pt')  # already the smallest
# Skip validation to speed up
results = model.train(..., val=False)
```

### Labels Not Found
```python
# Check structure
!ls -la /content/dataset/outputs/
!ls -la /content/dataset/outputs/labels/ | head -20
```

### Model Not Saving
```python
# Make sure Drive is mounted
drive.mount('/content/drive', force_remount=True)
```

---

## After Training

### Use the Model
```python
model = YOLO('/content/runs/detect/rack_detector/weights/best.pt')
results = model.predict('image.jpg')
```

### Improve Accuracy
- Generate more images (increase NUM_IMAGES)
- Train longer (increase epochs)
- Try larger model (yolov8s, yolov8m)
- Adjust augmentation parameters

### Deploy
```python
# Save for production
model.export(format='pt')  # PyTorch
model.export(format='tflite')  # Mobile
model.export(format='onnx')  # Cross-platform
```

---

## Performance Metrics to Monitor

After training, check:
- **mAP50**: Should be 60%+ with synthetic data
- **Precision**: How many predictions were correct
- **Recall**: How many objects were found
- **F1-Score**: Balance of precision and recall

View in TensorBoard:
```python
%load_ext tensorboard
%tensorboard --logdir /content/runs
```

---

Generated for synthetic rack dataset training
