# Write a function to read the image named " 0.png" in the current directory and display it using matplotlib. and show its shape.
import matplotlib.pyplot as plt
import cv2

def show_image_and_shape():
    # Read the image (note the leading space in the filename)
    img = cv2.imread("/home/amado/amado/CompVision/Lung-Tumor-Detection-and-Segmentation/playground/0.png")
    if img is None:
        print("Image not found.")
        return
    print("Image shape:", img.shape)
    # Convert BGR (OpenCV default) to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(" 0.png")
    plt.axis('off')
    plt.show()
    
show_image_and_shape()