"""
Reference: https://github.com/bupt-ai-cz/LLVIP/blob/main/toolbox/xml2txt_yolov5.py
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

def xml2yolov8_txt(xml_anno_path: Path, output_path: Path) -> None:
    root = ET.parse(str(xml_anno_path)).getroot()
    objects = root.findall('object')
    annotations = []
    for obj in objects:
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text.strip())
        xmax = int(bbox.find('xmax').text.strip())
        ymin = int(bbox.find('ymin').text.strip())
        ymax = int(bbox.find('ymax').text.strip())
        x_center = (0.5 * (xmin + xmax)) / 1280
        y_center = (0.5 * (ymin + ymax)) / 1024
        width = (xmax - xmin) / 1280
        height = (ymax - ymin) / 1024
        annotations.append(f"0 {x_center} {y_center} {width} {height}")
    with output_path.open('w') as f:
        f.write("\n".join(annotations))


def convert_batch(xml_anno_dir: Path, images_dir: Path, txt_output_dir: Path) -> None:
    if not txt_output_dir.exists():
        txt_output_dir.mkdir(parents=True)
    for p in tqdm(images_dir.glob("*.jpg")):
        xml_anno_path = xml_anno_dir / (p.stem + ".xml")
        output_path = txt_output_dir / (p.stem + ".txt")
        xml2yolov8_txt(xml_anno_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert xml annotations to yolov8 txt annotations")
    parser.add_argument("--xml-anno-dir", type=Path, default="datasets/LLVIP/Annotations", help="Path to xml annotations directory")
    parser.add_argument("--images-dir", type=Path, default="datasets/LLVIP/images/train", help="Path to images directory")
    parser.add_argument("--txt-output-dir", type=Path, default=None, help="Path to output directory")
    args = parser.parse_args()
    if args.txt_output_dir is None:
        # replace the last occurrence of "images" with "labels"
        args.txt_output_dir = Path("labels".join(str(args.images_dir).rsplit("images", 1)))
    return args


if __name__ == "__main__":
    args = parse_args()
    convert_batch(args.xml_anno_dir, args.images_dir, args.txt_output_dir)
