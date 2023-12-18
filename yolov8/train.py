import argparse

import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument("--checkpoint", type=str, default="yolov8n.pt", help="Path to detection model checkpoint")
    parser.add_argument("--resume", action="store_true", help="resume training from last checkpoint")
    parser.add_argument("--project", type=str, default="runs/train", help="Path to save the model")
    parser.add_argument("--data", type=str, default="LLVIP.yaml", help="Path to data config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on, either cpu or cuda (0, 1, 2, ...)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size, -1 for AutoBatch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lrf", type=float, default=0.1, help="final learning rate (lr * lrf)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help="learning optimizer")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.checkpoint, task='detect')

    # device
    device = torch.device(args.device)
    model.to(device)
    
    # train
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.lr,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        project=args.project,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
