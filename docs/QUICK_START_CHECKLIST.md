# Quick Reference Checklist

Print this and check off as you go!

---

## Phase 1: Cleanup (15 minutes)

- [ ] Read `ORGANIZATION_PLAN.md`
- [ ] Delete extra Blender installations:
  - [ ] `rm -rf blender_clean/`
  - [ ] `rm -rf blender_clean2/`
  - [ ] `rm -rf blender_custom/`
  - [ ] `rm -rf blender_custom_parent/`
  - [ ] `rm -rf blenderproc_blender/`
  - [ ] `rm -f blender-4.2.14-macos-arm64.dmg`
- [ ] Delete old outputs:
  - [ ] `rm -rf outputs_1/ v1/ v2/ newww/ Test_Image/`
- [ ] Delete old BlenderProc:
  - [ ] `rm -rf blenderproc_local/`
  - [ ] `rm -rf blender_custom_out_custom_packages/`
- [ ] Verify essential files remain:
  - [ ] `rack_gen.py`
  - [ ] `Assets/devices_front_only/` (with images)
  - [ ] `port_locations.json`
  - [ ] `blenderproc_packages/`

---

## Phase 2: Test Generation (10 minutes)

- [ ] Edit `rack_gen.py` - change `NUM_IMAGES = 1000` to `NUM_IMAGES = 10` (temporary test)
- [ ] Run:
```bash
env BLENDERPROC_CUSTOM_PACKAGES_PATH=/Users/aasritha/Downloads/Synthetic/blenderproc_packages \
    BLENDERPROC_SKIP_EMBEDDED_PIP_SETUP=1 \
    blenderproc run /Users/aasritha/Downloads/Synthetic/rack_gen.py
```
- [ ] Check output generated
- [ ] Verify files created:
  - [ ] `outputs/classes.txt` (should have 4 lines)
  - [ ] `outputs/synthetic_rack_*.png` (10 images)
  - [ ] `outputs/labels/*.txt` (10+ label files with augmentations)
- [ ] If test works → Go to Phase 3
- [ ] If test fails → Check error message, consult `DATASET_GENERATION_GUIDE.md`

---

## Phase 3: Full Generation (6-8 hours)

- [ ] Change `NUM_IMAGES` back to `1000` in `rack_gen.py`
- [ ] Delete test outputs: `rm -rf outputs/*`
- [ ] Run full generation:
```bash
env BLENDERPROC_CUSTOM_PACKAGES_PATH=/Users/aasritha/Downloads/Synthetic/blenderproc_packages \
    BLENDERPROC_SKIP_EMBEDDED_PIP_SETUP=1 \
    blenderproc run /Users/aasritha/Downloads/Synthetic/rack_gen.py
```
- [ ] Let it run overnight/in background
- [ ] Check progress with `=== Generating image X/1000 ===` messages
- [ ] Verify completion: Last message should be `DONE — all images generated.`

---

## Phase 4: Dataset Verification (5 minutes)

After generation completes:

- [ ] Count images: `ls -1 outputs/*.png | wc -l` (should be ~3,000)
- [ ] Count labels: `ls -1 outputs/labels/*.txt | wc -l` (should be ~3,000)
- [ ] Check classes file: `cat outputs/classes.txt`
  ```
  patch_panel
  pdu
  switch
  server
  ```
- [ ] Sample label file: `cat outputs/labels/synthetic_rack_000.txt`
  ```
  2 0.45 0.32 0.20 0.15
  0 0.65 0.50 0.18 0.10
  ```
- [ ] Dataset size: `du -sh outputs/` (should be ~2-3 GB)

---

## Phase 5: Upload to Google Drive (5 minutes)

- [ ] Zip dataset:
```bash
cd /Users/aasritha/Downloads/Synthetic
zip -r synthetic_dataset.zip outputs/
```
- [ ] Upload `synthetic_dataset.zip` to Google Drive
- [ ] Verify upload completed (check file size matches)
- [ ] Note the Drive path for Colab

---

## Phase 6: Prepare Colab Notebook (10 minutes)

- [ ] Open Google Colab: https://colab.research.google.com
- [ ] Create new notebook
- [ ] Rename: "Rack Detection Training"
- [ ] Have `COLAB_TRAINING_GUIDE.md` open
- [ ] Copy **Cell 1** into Colab (Install Requirements)
- [ ] Copy **Cell 2** into Colab (Mount Drive)
- [ ] Copy remaining cells one by one

---

## Phase 7: Run Training in Colab (1-4 hours)

- [ ] Run Cell 1: Install requirements ✓
- [ ] Run Cell 2: Mount Google Drive ✓
- [ ] Run Cell 3: Extract dataset ✓
- [ ] Run Cell 4: Create YOLO config ✓
- [ ] Run Cell 5: Prepare images and labels ✓
- [ ] **Run Cell 6: Train model** (will take 30 mins to 4 hours)
  - Monitor GPU usage: `!nvidia-smi`
  - Watch loss decrease
  - TensorBoard shows graphs
- [ ] Run Cell 7: Validate ✓
- [ ] Run Cell 8: Test on samples ✓
- [ ] Run Cell 9: Export model ✓
- [ ] Run Cell 10: Make predictions ✓
- [ ] Run Cell 11: Download results ✓

---

## Phase 8: Use Your Model (5 minutes)

- [ ] Download `best.pt` from Colab
- [ ] Use in your application:

```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('your_rack_image.jpg')
```

- [ ] Export to other formats if needed:
  - `model.export(format='onnx')` - Cross-platform
  - `model.export(format='tflite')` - Mobile
  - `model.export(format='coreml')` - iOS

---

## 🆘 Quick Fixes

**Generation script won't run:**
- [ ] Is Blender installed? → Check path
- [ ] Do `Assets/devices_front_only/` have images?
- [ ] Is `blenderproc_packages/` present?

**Out of memory during generation:**
- [ ] Close other apps
- [ ] Reduce `imgsz` in script
- [ ] Try again

**Colab training is slow:**
- [ ] Check GPU: `!nvidia-smi` shows GPU?
- [ ] Use smaller model: `yolov8n.pt` (already default)
- [ ] Reduce batch size to 8

**Labels not found in Colab:**
- [ ] Check Drive is mounted
- [ ] Verify zip extracted correctly
- [ ] Check folder structure: `!ls -la /content/dataset/outputs/`

---

## 📊 Expected Results

### After Generation
```
outputs/
├── classes.txt
├── synthetic_rack_000.png ... synthetic_rack_2999.png
├── labels/
│   ├── synthetic_rack_000.txt ... synthetic_rack_2999.txt
└── (Total: ~3,000 images + 3,000 labels)
```

### After Training
```
Model saved to: /content/runs/detect/rack_detector/weights/best.pt
Results:
- mAP50: 0.65+ (65%+ accuracy)
- Precision: 0.70+
- Recall: 0.60+
```

### After Export
```
best.pt         # PyTorch format
best.onnx       # Cross-platform
best.tflite     # Mobile
best.mlmodel    # iOS
```

---

## 📞 When You Get Stuck

| Problem | Solution |
|---------|----------|
| Script won't start | Check `DATASET_GENERATION_GUIDE.md` Troubleshooting |
| Wrong labels | Script auto-handles 3D→2D projection, check camera |
| Slow generation | Normal - Blender rendering is slow. Use GPU. |
| Colab training error | Check `COLAB_TRAINING_GUIDE.md` Troubleshooting |
| Model accuracy low | Generate more images or train longer |
| Want to improve | See "Training Tips" in `COLAB_TRAINING_GUIDE.md` |

---

## Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Cleanup | 15 min |
| 2 | Test generation | 10 min |
| 3 | Full generation | 6-8 hours ⏰ |
| 4 | Verification | 5 min |
| 5 | Upload to Drive | 5 min |
| 6 | Colab setup | 10 min |
| 7 | Training | 1-4 hours ⏰ |
| 8 | Use model | 5 min |
| **TOTAL** | | **7-12 hours** |

⏰ = Can run while you sleep!

---

## Success Criteria

✅ **You're done when:**
- [ ] 3,000 images with labels in `outputs/`
- [ ] Model trained in Colab with mAP50 > 0.60
- [ ] Can run predictions on new rack images
- [ ] Model exported and ready to use

---

## Backup Plan

If something goes wrong:

1. **Partial dataset?** Train with what you have, results will be lower
2. **Colab timeout?** Restart and continue from last checkpoint
3. **Bad generation?** Start over - script is idempotent (safe to re-run)
4. **Model not converging?** Try:
   - Train longer (200 epochs)
   - Use larger model (yolov8s instead of yolov8n)
   - Generate more images (2,000 instead of 1,000)

---

## Done! 🎉

Once you complete all phases, you'll have:
- ✅ 3,000 synthetic training images
- ✅ Automatic YOLO-format labels
- ✅ Trained YOLOv8 object detection model
- ✅ Model ready for production use

**Next:** Deploy your model to detect server rack components in real world images!

---

Print this checklist and keep it nearby! ✓

