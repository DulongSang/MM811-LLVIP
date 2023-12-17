import argparse

import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument("--load-pretrained", type=str, default=None, help="Path to the pretrained detection model checkpoint")
    parser.add_argument("--data", type=str, default="LLVIP.yaml", help="Path to data config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on, either cpu or cuda (0, 1, 2, ...)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.data, task='detect')

    # device
    device = torch.device(args.device)
    model.to(device)

    # load pretrained weights
    if args.load_pretrained:
        model.load(args.load_pretrained)
    
    # train
    model.train(data=args.data, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
