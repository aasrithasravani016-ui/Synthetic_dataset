import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a YOLOv8n POC model on the Dataset folder.'
    )
    parser.add_argument(
        '--dataset-dir',
        default=os.path.dirname(os.path.abspath(__file__)),
        help='Path to the Dataset folder containing data.yaml, images/, labels/'
    )
    parser.add_argument(
        '--model',
        default='yolov8n.pt',
        help='Ultralytics YOLO model checkpoint to start from.'
    )
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size.')
    parser.add_argument('--project', default='runs/train_poc', help='Output project folder.')
    parser.add_argument('--name', default='rack_poc', help='Run name inside project.')
    parser.add_argument('--device', default='0', help='Training device, e.g. 0, cpu, cuda:0.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    data_yaml = os.path.join(dataset_dir, 'data.yaml')

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"data.yaml not found in dataset dir: {dataset_dir}\n"
            "Make sure the Dataset folder contains data.yaml, images/, and labels/."
        )

    print(f"Training with dataset: {dataset_dir}")
    print(f"Using config: {data_yaml}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")

    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        exist_ok=True,
    )

    print("Training complete.")
    print(f"Best weights and logs available at: {os.path.join(args.project, args.name)}")


if __name__ == '__main__':
    main()
