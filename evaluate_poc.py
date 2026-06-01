import argparse
import os
import random
import glob
from pathlib import Path
from collections import Counter

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLO POC dataset and model.')
    parser.add_argument('--dataset-dir', default='.', help='Path to Dataset folder.')
    parser.add_argument('--model', default=None, help='Path to trained YOLO model weights (best.pt).')
    parser.add_argument('--output-dir', default='evaluate_results', help='Directory to save reports and images.')
    parser.add_argument('--create-splits', action='store_true', help='Create train/val/test text splits if not present.')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio when creating new splits.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio when creating new splits.')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test split ratio when creating new splits.')
    parser.add_argument('--split-seed', type=int, default=42, help='Random seed used for any split creation.')
    parser.add_argument('--eval-split', default='val', choices=['train', 'val', 'test', 'all'], help='Which split to evaluate when model is provided.')
    parser.add_argument('--sample-count', type=int, default=8, help='Number of sample images to run inference on.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation and inference.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for validation.')
    parser.add_argument('--device', default='0', help='Device for model evaluation, e.g. 0, cpu, cuda:0.')
    return parser.parse_args()


def read_data_yaml(data_yaml_path):
    names = []
    with open(data_yaml_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('names:'):
                try:
                    rest = line.split(':', 1)[1].strip()
                    if rest.startswith('['):
                        names = [n.strip().strip("'\"") for n in rest.strip('[]').split(',') if n.strip()]
                except Exception:
                    pass
            elif line.startswith('names') and ':' not in line:
                # multiline names block is not supported in this script
                continue
    return names


def gather_dataset_stats(dataset_dir):
    images_dir = Path(dataset_dir) / 'images'
    labels_dir = Path(dataset_dir) / 'labels'
    image_paths = sorted(images_dir.glob('*.png')) + sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.jpeg'))
    image_paths = sorted(set(image_paths))
    label_paths = sorted(labels_dir.glob('*.txt'))

    image_basenames = {p.stem for p in image_paths}
    label_basenames = {p.stem for p in label_paths}

    matched_images = sorted([p for p in image_paths if p.stem in label_basenames])
    missing_labels = sorted([p for p in image_paths if p.stem not in label_basenames])
    orphan_labels = sorted([p for p in label_paths if p.stem not in image_basenames])

    annotations = []
    class_counter = Counter()
    per_image_counts = {}

    for label_path in label_paths:
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        per_image_counts[label_path.stem] = len(lines)
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                class_counter[int(parts[0])] += 1
        annotations.extend(lines)

    return {
        'images_total': len(image_paths),
        'label_files_total': len(label_paths),
        'images_with_labels': len(matched_images),
        'missing_labels': missing_labels,
        'orphan_labels': orphan_labels,
        'class_counts': class_counter,
        'per_image_counts': per_image_counts,
        'image_paths': image_paths,
        'label_paths': label_paths,
    }


def write_split_files(dataset_dir, image_names, ratios, seed):
    random.Random(seed).shuffle(image_names)
    n = len(image_names)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    splits = {
        'train': image_names[:train_end],
        'val': image_names[train_end:val_end],
        'test': image_names[val_end:],
    }
    splits_dir = Path(dataset_dir) / 'splits'
    splits_dir.mkdir(exist_ok=True)
    for split_name, names in splits.items():
        with open(splits_dir / f'{split_name}.txt', 'w') as f:
            for name in names:
                f.write(name + '\n')
    return splits


def load_split(dataset_dir, split_name):
    path = Path(dataset_dir) / 'splits' / f'{split_name}.txt'
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def build_report(args, stats, split_info, eval_metrics=None, sample_images=None):
    report_path = Path(args.output_dir) / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write('# YOLO POC Dataset and Model Evaluation Report\n\n')
        f.write('## Dataset Summary\n\n')
        f.write(f'- Dataset directory: `{args.dataset_dir}`\n')
        f.write(f'- Total images: {stats["images_total"]}\n')
        f.write(f'- Total label files: {stats["label_files_total"]}\n')
        f.write(f'- Images with labels: {stats["images_with_labels"]}\n')
        f.write(f'- Missing label files: {len(stats["missing_labels"])}\n')
        f.write(f'- Orphan label files: {len(stats["orphan_labels"])}\n\n')

        f.write('### Split summary\n\n')
        if split_info:
            for split_name, values in split_info.items():
                f.write(f'- {split_name}: {len(values)} images\n')
        else:
            f.write('- No split files found. Use `--create-splits` to generate train/val/test splits.\n')
        f.write('\n')

        f.write('### Annotation counts per class\n\n')
        if stats['class_counts']:
            for class_id, count in stats['class_counts'].items():
                f.write(f'- Class {class_id}: {count} annotations\n')
        else:
            f.write('- No annotations found.\n')
        f.write('\n')

        if eval_metrics is not None:
            f.write('## Model Evaluation Metrics\n\n')
            f.write(f'- Evaluated split: `{args.eval_split}`\n')
            for key, value in eval_metrics.items():
                f.write(f'- {key}: {value}\n')
            f.write('\n')

        if sample_images:
            f.write('## Sample Inference Results\n\n')
            f.write('See the `predictions/` folder for annotated prediction images.\n\n')
            f.write('### Sample image list\n\n')
            for sample_image in sample_images:
                f.write(f'- {sample_image}\n')
            f.write('\n')

        f.write('## Notes\n\n')
        f.write('- This report was generated by `Dataset/evaluate_poc.py`.\n')
        f.write('- If no validation split exists, evaluation uses the full dataset.\n')

    return report_path


def run_model_eval(args, data_yaml):
    if YOLO is None:
        raise RuntimeError('ultralytics package is not installed. Install with `pip install ultralytics`.')
    model = YOLO(args.model)
    print(f'Running model evaluation on {args.eval_split} split...')
    metrics = model.val(data=data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device)
    result = {
        'mAP50': metrics.metrics[0].get('mAP50', 'n/a') if hasattr(metrics, 'metrics') else str(metrics),
        'mAP50-95': metrics.metrics[0].get('mAP50-95', 'n/a') if hasattr(metrics, 'metrics') else str(metrics),
        'precision': metrics.metrics[0].get('precision', 'n/a') if hasattr(metrics, 'metrics') else str(metrics),
        'recall': metrics.metrics[0].get('recall', 'n/a') if hasattr(metrics, 'metrics') else str(metrics),
    }
    return model, result


def run_sample_predictions(args, model, sample_paths):
    target_dir = Path(args.output_dir) / 'predictions'
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f'Running predictions for {len(sample_paths)} sample images...')
    results = model.predict(source=sample_paths, imgsz=args.imgsz, device=args.device, save=True, save_txt=False, project=args.output_dir, name='predictions')
    saved = []
    for res in results:
        saved.append(str(res.path))
    return saved


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_yaml = Path(dataset_dir) / 'data.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f'data.yaml not found in {dataset_dir}')

    stats = gather_dataset_stats(dataset_dir)
    split_info = {}

    if args.create_splits:
        image_names = [p.stem for p in stats['image_paths']]
        split_info = write_split_files(dataset_dir, image_names, (args.train_ratio, args.val_ratio, args.test_ratio), args.split_seed)
        print('Created split files in Dataset/splits/')
    else:
        for split_name in ['train', 'val', 'test']:
            split_values = load_split(dataset_dir, split_name)
            if split_values is not None:
                split_info[split_name] = split_values

    eval_metrics = None
    sample_results = None
    if args.model:
        if YOLO is None:
            raise RuntimeError('ultralytics is required to evaluate a model. Install with pip install ultralytics.')

        eval_yaml = str(data_yaml)
        if not split_info and args.eval_split != 'all':
            print('No split files found; evaluation will use full dataset.')
        model, eval_metrics = run_model_eval(args, eval_yaml)

        sample_images = stats['image_paths'][:args.sample_count]
        sample_results = run_sample_predictions(args, model, [str(p) for p in sample_images])

    report_path = build_report(args, stats, split_info, eval_metrics, sample_results)
    print(f'Report written to {report_path}')


if __name__ == '__main__':
    main()
