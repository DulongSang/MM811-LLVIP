"""
Reference: https://github.com/bupt-ai-cz/LLVIP/blob/main/yolov5/val.py
Simplified version of the original script
"""

import argparse

import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to detection model checkpoint")
    parser.add_argument("--data", type=str, default="LLVIP.yaml", help="Path to data config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on, either cpu or cuda (0, 1, 2, ...)")
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--conf', type=float, default=0.6, help='confidence threshold')
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model, task='detect')

    # device
    device = torch.device(args.device)
    model.to(device)

    # half
    use_half = (device.type != 'cpu') and args.half  # half precision only supported on CUDA
    model.half() if use_half else model.float()

    model.val(data=args.data, conf=args.conf, classes=[0], save_json=True)


if __name__ == "__main__":
    main()
