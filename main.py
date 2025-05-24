# main.py

import argparse
import os
import torch
import logging
from train.utils import preprocess_image, read_image, visualize_segmentation_overlay
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
    parser.add_argument('--output_name', type=str, default='output.png', help='Name to save segmentation output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse and return arguments
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.mode == 'infer':
        logging.info("Running in inference mode.")
        # assert args.checkpoint is not None, "Please provide --checkpoint for inference mode"
        assert args.input_image is not None, "Please provide --input_image for inference mode"
        
        logging.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = get_model(args.model)
        
        if args.checkpoint is None:
            if args.model == "unet":
                args.checkpoint = "experiments/unet/best_unetv2.pth"
            elif args.model == "unetpp":
                args.checkpoint = "experiments/unet++/best_unetpp_extra(v3).pth"
            elif args.model == "attention_unet":
                args.checkpoint = "experiments/attention unet/best_unetattn(v2).pth"
        else:
            if args.model == "unet":
                args.checkpoint = "experiments/unet/"+args.checkpoint
            elif args.model == "unetpp":
                args.checkpoint = "experiments/unet++/"+args.checkpoint
            elif args.model == "attention_unet":
                args.checkpoint = "experiments/attention unet/"+args.checkpoint                
                
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

        logging.info(f"Saving output mask to: outputs/{args.output_name}")
        
        os.makedirs("outputs", exist_ok=True)
        visualize_segmentation_overlay(img, mask, output_path="outputs/"+args.output_name)        
        return

if __name__ == "__main__":
    main()