from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from automatic_mask_and_probability_generator import SamAutomaticMaskAndProbabilityGenerator
import os


def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)
    return image


def process(sourcedir, dst, sam_checkpoint, model_type="vit_h"):
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    generator = SamAutomaticMaskAndProbabilityGenerator(sam)

    save_root_dir = dst

    for i, filename in enumerate(os.listdir(sourcedir)):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            generator = SamAutomaticMaskAndProbabilityGenerator(sam)
            print(f'{i}th pic: {filename}')

            img_path = os.path.join(sourcedir, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = generator.generate(image)

            p_max = None
            for mask in masks:
                p = mask["prob"]
                if p_max is None:
                    p_max = p
                else:
                    p_max = np.maximum(p_max, p)

            sobel_max = None
            for mask in masks:
                p = mask["sobel"]
                if sobel_max is None:
                    sobel_max = p
                else:
                    sobel_max = np.maximum(sobel_max, p)

            semantic_map = normalize_image(p_max)
            edges = normalize_image(sobel_max)

            os.makedirs(save_root_dir, exist_ok=True)
            save_semantic_path = os.path.join(save_root_dir, f'{os.path.splitext(filename)[0]}_semantic.png')
            save_edge_path = os.path.join(save_root_dir, f'{os.path.splitext(filename)[0]}_edge.png')
            plt.imsave(save_edge_path, edges)
            plt.imsave(save_semantic_path, semantic_map)
            del masks, p_max, sobel_max, p, edges, semantic_map, generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SAM edge and semantic priors for images')
    parser.add_argument('--src', type=str, required=True, help='Source directory containing images')
    parser.add_argument('--dst', type=str, required=True, help='Destination directory for SAM priors')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint file')
    parser.add_argument('--model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type: vit_h (huge), vit_l (large), or vit_b (base)')

    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise ValueError(f"Source directory does not exist: {args.src}")

    print(f"Processing images from: {args.src}")
    print(f"Saving results to: {args.dst}")
    print(f"Using SAM checkpoint: {args.sam_checkpoint}")
    print(f"Model type: {args.model_type}")

    process(args.src, args.dst, args.sam_checkpoint, args.model_type)

    print("Done!")
