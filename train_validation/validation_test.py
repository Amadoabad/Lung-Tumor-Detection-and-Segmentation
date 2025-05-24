import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def test_detection(model, dataset_dir, num_samples=30):
    """
    Test detection on some validation images and visualize results
    """

    import random

    images_dir = os.path.join(dataset_dir, 'images')
    all_image_files = os.listdir(images_dir)

    # Randomly select num_samples without replacement
    image_files = random.sample(all_image_files, min(num_samples, len(all_image_files)))

    plt.figure(figsize=(30, 50))

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)

        # Run detection
        results = model(img_path)
        result_img = results[0].plot()

        plt.subplot(num_samples, 1, i+1)
        plt.imshow(result_img)
        plt.title(f"Sample {i+1}: {img_file}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('detection_results.png')
    plt.show()  # Show the plot in Colab



model_path= 'weights/best_small_object_detector.pt'
yolo_model = YOLO(model_path)
# Test detection on some validation images
test_detection(yolo_model, "Dataset/val/")
