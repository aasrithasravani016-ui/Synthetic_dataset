import argparse
import os
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a stable YOLOv8 model on the New_Dataset folder.'
    )
    parser.add_argument(
        '--dataset-dir',
        default=os.path.dirname(os.path.abspath(__file__)),
        help='Path to the New_Dataset folder containing data.yaml, images/, and labels/'
    )
    parser.add_argument(
        '--model',
        default='yolov8n.pt',
        help='Ultralytics YOLO model checkpoint to start from.'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size.')
    parser.add_argument('--project', default='runs/train_stable', help='Output project folder.')
    parser.add_argument('--name', default='rack_stable', help='Run name inside project.')
    parser.add_argument('--device', default='0', help='Training device, e.g. 0 or cpu.')
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer for training.')
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay.')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience.')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    data_yaml = os.path.join(dataset_dir, 'data.yaml')

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"data.yaml not found in dataset dir: {dataset_dir}\n"
            "Make sure the New_Dataset folder contains data.yaml, images/, and labels/."
        )

    print(f"Training with dataset: {dataset_dir}")
    print(f"Using config: {data_yaml}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")
    print(f"Optimizer: {args.optimizer}, lr0: {args.lr0}, lrf: {args.lrf}, weight_decay: {args.weight_decay}, patience: {args.patience}")

    if args.device == '0' and torch is not None and not torch.cuda.is_available():
        print("CUDA is not available; switching device to 'cpu'.")
        args.device = 'cpu'

    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,
    )

    print('Training complete.')
    print(f"Best weights and logs available at: {os.path.join(args.project, args.name)}")


if __name__ == '__main__':
    main()
