# Road Signs Shapes Detection with OpenCV

This repository contains the solution for detecting shapes of different road signs in croppped images. Detection is normally performed by training deep learning models but this repo is pure Image Processing using openCV operations which has advantage over traning a deep learning model regarding speed and dataset availability. The solution has classified correctly all 35 images with 7 different shapes and 5 images each. The shapes are:
1. Square
2. Horizontal Rectangle
3. Vertical Rectangle
4. Diamond
5. Pentagon
6. Octagon
7. Circle


## Working

The solution adopts two basic techniques for deriving the final results with extension of other image processing operations.
1. Canny Edge Detection
2. Color based Segmentation
3. Bounding Box Prediction

First it runs the Canny Edge Detection to detect all the sharp edges in an image. Then it extracts the edges which are closed only and removing all others including lines and curves (a road sign shape is always a closed figure). After that, it considers the closed shapes which have area greater than 30% of the whole image area (a sign in cropped image contains larger portion of the image). This filter removes all small shapes which are not required including text inside the sign. We then compute a ratio r of the perimeter of a shape to its area and considering the one with smallest ratio. This technique removes all the irregular shapes as shown below:

<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/_ratio.png" width=1000/>

Once the desired shape is extracted, finally we apply some contour approximation methos inlcuding Convex Hull and approxPolyDP to first fill out any broken pieces of shape and then finding the number of vertices which gives us the final shape.

Color based Segmentation is used when the Canny Edge Detection fails to extract the desired contours (in some cases, the image is zoomed and the borders of signs are cropped). In this case, we go for color based segmentation which extracts the dominant color in the image (probably the road sign color). We then compute the contours and repeat the above steps.

Third, even if color based segmentation fails to work, we then compute bounding box prediction on all contours (small in area) and compute a box around them. This box is then checked whether the shape is sqaure, rectangle or diamond.

## Gallery

<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/sample_1.png" width=1000/>
<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/sample_2.png" width=1000/>
<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/sample_3.png" width=1000/>
<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/sample_4.png" width=1000/>
<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/sample_5.png" width=1000/>
<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/sample_6.png" width=1000/>
<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/sample_7.png" width=1000/>


## How to run the Code
Clone the repository and head over to the root directory. Enter the following command with two flags to specify.
    
    python run.py --images_path=images --verbose=1
    
### Flags:
1. **--images_path** (default ./images) : specify the path to the images directory. The direcotry must contain images.
2. **--verbose** (default 1): specify the verbosity level. Either 1 or 0. Prints out the shapes in terminal when verbose=0 else visualize the steps involved while processing the image graphically.
 
 
 # References
 
 Colors Quantization (KMeans Clustering)
 https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
 
 Shape detection with OpenCV
 https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
 
 Contours Features
 https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
 
 Color Based Segmentation
 https://realpython.com/python-opencv-color-spaces/
