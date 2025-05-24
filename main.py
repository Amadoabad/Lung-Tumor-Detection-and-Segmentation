import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.predict_inference import visualize_single_image_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Mask R-CNN tumor prediction on a single image.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model weights (.pth)")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image to be tested")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Score threshold for displaying predictions")

    args = parser.parse_args()

    visualize_single_image_prediction(args.model_path, args.image_path, args.score_threshold)