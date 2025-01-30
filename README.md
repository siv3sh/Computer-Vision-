
# Image Processing and Display Using Various Python Libraries

## Introduction
This document explains how to read, write, and display images using various Python libraries including OpenCV, ImageIO, Matplotlib, PIL, and Scikit-Image. These libraries offer diverse functionalities for image processing, computer vision, and visualizations.

---

## 1. Reading and Displaying Image Using OpenCV
OpenCV (Open Source Computer Vision Library) is a popular library for fast and efficient image processing.

### Key Features:
- Image transformations (rotation, resizing, perspective change)
- Feature detection (Harris Corner, SIFT, SURF)
- Object and motion detection
- Filtering (blur, sharpening, edge detection)

### Code Example:
```python
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('sampleimg.jpg')
plt.imshow(img)
print(img)
```

### Notes:
- OpenCV stores images in BGR format by default.
- Flags for image reading:
  - `cv2.IMREAD_COLOR`: Loads a color image.
  - `cv2.IMREAD_GRAYSCALE`: Loads an image in grayscale.
  - `cv2.IMREAD_UNCHANGED`: Loads an image with its alpha channel.

---

## 2. Reading and Writing Image Using ImageIO
ImageIO is a library used for reading and writing images in various formats.

### Code Example:
```python
import imageio as iio

# Read an image
img1 = iio.imread("sampleimg.jpg")

# Write it in a new format
iio.imwrite("g4g.png", img1)
```

---

## 3. Reading and Displaying Image Using Matplotlib
Matplotlib is a versatile library for creating plots and visualizations, including image display.

### Code Example:
```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Read Image
img2 = mpimg.imread('g4g.png')

# Display Image
plt.imshow(img2)
plt.show()
```

### Notes:
- OpenCV uses BGR format, while Matplotlib uses RGB.

---

## 4. Reading and Displaying Image Using PIL
PIL (Python Imaging Library) offers basic image processing capabilities.

### Code Example:
```python
from PIL import Image

# Read image
img3 = Image.open('g4g.png')

# Display Image
img3.show()

# Image Information
print("Format: ", img3.format)
print("Mode: ", img3.mode)
print("Filename: ", img3.filename)
print("Size: ", img3.size)
```

---

## 5. Reading and Displaying Image Using Scikit-Image
Scikit-Image is an advanced library for image processing.

### Key Features:
- Filters (Gaussian, Sobel, Prewitt)
- Feature extraction (edges, corners, blobs)
- Segmentation (watershed, thresholding)

### Code Example:
```python
from skimage import io, filters

# Load grayscale image
image = io.imread("g4g.png", as_gray=True)

# Display Image
io.imshow(image)
io.show()
```

---

## 6. Accessing Pixel Values Using OpenCV
You can access and manipulate individual pixel values using OpenCV.

### Code Example:
```python
import cv2
import numpy as np

# Load image
img4 = cv2.imread('sampleimg.jpg')

# Access pixel value
pixel = img4[100, 100]
print("Pixel Value: ", pixel)

# Access only blue pixel
blue = img4[100, 100, 0]
print("Blue Pixel Value: ", blue)
```

---

## 7. Converting Image Formats
OpenCV supports reading and writing the following image formats:
- Windows bitmaps (`*.bmp`, `*.dib`)
- JPEG files (`*.jpeg`, `*.jpg`, `*.jpe`)
- PNG files (`*.png`)
- TIFF files (`*.tiff`, `*.tif`)

### Code Example:
```python
# Load and save an image in a different format
img5 = cv2.imread('sampleimg.jpg')
cv2.imwrite("newimg5.png", img5)
```

---

## Conclusion
This document provides an overview of various Python libraries for reading, writing, and displaying images, along with code examples and key functionalities. Each library has unique features, making them suitable for different image processing and computer vision tasks.
