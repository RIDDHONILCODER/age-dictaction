import cv2
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------
# Define a utility function to display images using Matplotlib.
# 1. Set up the figure size.
# 2. Check if image is grayscale or color.
# 3. Convert color images from BGR to RGB for correct rendering.
# 4. Set the plot title and hide the axis.
# 5. Display the image on the screen.
# --------------------------------------------------------------
def displayimage(title,image):
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


    
image=cv2.imread('lambo.jpg')
grayimage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
displayimage("Gray scale image",grayimage)

print("Select an option:")
print("1. Sobel Edge Detection")
print("2. Canny Edge Detection")
print("3. Laplacian Edge Detection")
print("4. Gaussian Smoothing")
print("5. Median Filtering")
print("6. Exit")

while True:
    choice=input("enter your choice")
# --------------------------------------------------------------
# Sobel Edge Detection:
# 1. Calculate Sobel filters along the x and y directions.
# 2. Convert both results to 8-bit images.
# 3. Combine them using bitwise OR.
# 4. Display the combined edge map.
# --------------------------------------------------------------
    if choice=="1":
        #sobel edge - sobelx = cv2.Sobel(src, ddepth, dx, dy, ksize=3)
        sobelx=cv2.Sobel(grayimage,cv2.CV_64F,1,0,ksize=3)
        sobely=cv2.Sobel(grayimage,cv2.CV_64F,0,1,ksize=3)
        combindsobel=cv2.bitwise_or(sobelx.astype(np.uint8),sobely.astype(np.uint8))
        displayimage("sobel edge dictaction ",combindsobel)
    # --------------------------------------------------------------
# Canny Edge Detection:
# 1. Ask for lower and upper thresholds.
# 2. Apply Canny edge detection, which:
#    - Smooths the image with a Gaussian filter.
#    - Finds intensity gradients.
#    - Applies non-maximum suppression.
#    - Uses double-thresholding and edge tracking.
# 3. Display the detected edges.
# --------------------------------------------------------------
    if choice=="2":
        lt=int(input("enter lower threshold"))
        ut=int(input("enter uper threshold"))
        canyimage=cv2.Canny(grayimage,lt,ut)
        displayimage("canny edge dictation ",canyimage)
    # --------------------------------------------------------------
# Laplacian Edge Detection:
# 1. Apply the Laplacian operator (second derivative).
# 2. Take the absolute value of the result to handle negative gradients.
# 3. Convert to 8-bit for display.
# 4. Show the resulting edges.
# --------------------------------------------------------------
    if choice=="3":
        lapimage=cv2.Laplacian(grayimage,cv2.CV_64F)
        displayimage("laplacian edge dictation ",np.abs(lapimage).astype(np.uint8))