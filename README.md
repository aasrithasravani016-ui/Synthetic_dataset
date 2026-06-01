Dataset copy for POC training.

Contents:
- images/: all generated PNG images
- labels/: YOLO-format TXT labels
- classes.txt: class names mapping
- data.yaml: YOLO dataset config for ultralytics

POC training command:
  yolo task=detect mode=train model=yolov8n.pt data=Dataset/data.yaml epochs=10 imgsz=640 batch=8 project=runs/train_poc name=rack_poc

Inference command:
  yolo detect predict model=runs/train_poc/rack_poc/weights/best.pt source=Dataset/images/synthetic_rack_000.png

Colab training example:
  !pip install ultralytics
  !python Dataset/train_poc.py --epochs 10 --batch 8 --imgsz 640

Evaluation and results reporting:
  !python Dataset/evaluate_poc.py --model runs/train_poc/rack_poc/weights/best.pt --output-dir Dataset/evaluate_results

Full POC documentation:
  See `Dataset/POC_REPORT.md` for generation details, augmentation techniques, model selection, dataset statistics, and training/evaluation workflow.

If you want a faster proof-of-concept run, use `--epochs 5`.
