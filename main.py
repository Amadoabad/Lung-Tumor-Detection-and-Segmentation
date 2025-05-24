# main.py

import argparse
import os
import torch
import logging
from train.utils import preprocess_image, read_image
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def get_model(model_name):
    model_name = model_name.lower()
    
    if model_name == "unet":
        from models.unet import UNet
        logging.info("Using UNet model.")
        return UNet()
    
    elif model_name == "unetpp":
        from models.unetpp import UNetPlusPlus
        logging.info("Using UNet++ model.")
        return UNetPlusPlus()
    
    elif model_name == "attention_unet":
        from models.attention_unet import AttentionUNet
        logging.info("Using Attention UNet model.")
        return AttentionUNet()
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Produce segmentation for a lung image")
    parser.add_argument('--model', type=str, default='unet', help='Model name: unet, unetpp, attention_unet')
    parser.add_argument('--mode', type=str, default='infer', choices=['train', 'infer'], help='Mode: train or infer')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--input_image', type=str, help='Path to input image for inference')
    parser.add_argument('--output_path', type=str, default='output.png', help='Path to save segmentation output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse and return arguments
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.mode == 'infer':
        logging.info("Running in inference mode.")
        assert args.checkpoint is not None, "Please provide --checkpoint for inference mode"
        assert args.input_image is not None, "Please provide --input_image for inference mode"
        
        logging.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = get_model(args.model)
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        model.to(args.device)
        model.eval()

        import torchvision.transforms as T
        import matplotlib.pyplot as plt
        import torch.nn.functional as F


        logging.info(f"Loading input image: {args.input_image}")
        img = read_image(args.input_image)
        input_tensor = preprocess_image(img)

        logging.info("Performing inference...")
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)
            mask = output.squeeze().cpu().numpy()

        logging.info(f"Saving output mask to: {args.output_path}")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(mask, cmap='gray')
        plt.savefig(args.output_path)
        plt.show()
        return

if __name__ == "__main__":
    main()