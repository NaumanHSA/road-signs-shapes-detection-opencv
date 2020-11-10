"""
Code by Muhammad Nouman Ahsan(Original Author)

Simple Shape Detection with Image Processing using OpenCV
IDE: Pycharm(you can use it in other IDEs)

"""

# Convenient way for choosing right Contour to Detect Road Signs Shapes
#
# Algorithm:
# Reading RGB image --> Detect Contours --> Choose the best one --> Predict Shape
# if fails to choose the best contour then
# Reading RGB image --> Segment Region based on Color --> Detect Contours --> Predict Shape

# Result:
# Shape Prediction : All Steps Visualized if verbose = 1
# Shape Prediction : Only print the Shape Prediction if verbose = 0

# import necessary packages and ShapeClassifier Class
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np


class ShapeClassifier:
    """
       A class used to detect road sings shapes
       ...
       Attributes
       ----------
       verbose : int
           a integer values (0, 1) for results verbosity
       image_path : str
           stores the path of the image currently processed
       colors : dict
           a dict object containing upper and lower boundaries for different colors
           used in color based segmentation

       Methods
       -------
       _countors_based_segmentation(self, _image, contours, flag)
           process contours and predicts the final result

       color_based_segmentation(self, image)
            preforms color based segmentation in case contours based detection fails

       _detect_shape_contours_approx(self, cnt)
            detects the shape by applying cv2.approxPolyDP() for contours approximation

        box_prediction(self, contours)
            in case contours based segmentation fails, this method combines small contours to
            predict a single box for shape prediction

        display_contours_restuls(self, image, cnts, extracted_cnts, processed_cnt, shape)
            display resuls

        display_box_results(self, image, contours, params)
            display results

        preprcess_image(self, _image, n=8)
            applyies K-Mean Clustering algorithm to perform color quantization for better
            color based segmentation

        predict_shape(self, image_path)
            first method to call which act as init method
       """

    # initialize important variables
    def __init__(self, verbose=1):
        self.verbose = verbose
        self.image_path = None
        # color boundaries used to perform color based segmentation
        self.colors = {
            "low_red": np.array([0, 193, 110]),
            "high_red": np.array([185, 255, 255]),
            "low_blue": np.array([110, 50, 50]),
            "high_blue": np.array([130, 255, 255]),
            "low_green": np.array([25, 80, 80]),
            "high_green": np.array([85, 255, 255]),
            "low_yellow": np.array([14, 85, 124]),
            "high_yellow": np.array([45, 255, 255]),
        }

    def _countors_based_segmentation(self, _image, contours, flag):
        # compute image area
        _area_image = _image.shape[0] * _image.shape[1]
        # seperate contours with area greater than 30% of image area
        _large_area_contours = [cnt for cnt in contours if cv2.contourArea(cnt) / _area_image > 0.3]

        # if no contours has area greater than 30% of image area
        if len(_large_area_contours) == 0:
            if flag == 0:
                # perform color based segmentation
                self.color_based_segmentation(_image)
            else:
                # perform box prediction using small contours
                box_params = self.box_prediction(contours)
                if self.verbose == 1:
                    # display results
                    self.display_box_results(_image, contours, box_params)
                else:
                    if len(box_params) > 0:
                        print("{}: {}".format(self.image_path, box_params[4]))
            return

        # extract the contour having smallest ratio of its perimeter to its area
        # removes contours with irregular shapes
        min_ratio = min([cv2.arcLength(cnt, True) / cv2.contourArea(cnt) for cnt in _large_area_contours])
        max_cnt = [cnt for cnt in _large_area_contours
                   if cv2.arcLength(cnt, True) / cv2.contourArea(cnt) == min_ratio][0]

        # draw all contours
        mask_cnt_all = np.zeros((_image.shape[0], _image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask_cnt_all, contours, -1, 255, thickness=cv2.FILLED)

        # draw the one extracted before
        mask_cnt = np.zeros((_image.shape[0], _image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask_cnt, [max_cnt], -1, 255, thickness=cv2.FILLED)

        # do some morphological operations to remove small regions
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_cnt = cv2.erode(mask_cnt, kernel, mask_cnt, iterations=1)

        # again find contours and extract the one with largest area
        contours, hierarchy = cv2.findContours(mask_cnt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = max(contours, key=cv2.contourArea)

        # perform convextHull operation which fills the holes inside a broken shape
        mask_covex = np.zeros((_image.shape[0], _image.shape[1]), dtype=np.uint8)
        max_cnt_convex = cv2.convexHull(max_cnt)
        cv2.drawContours(mask_covex, [max_cnt_convex], -1, 255, thickness=cv2.FILLED)
        shape = self._detect_shape_contours_approx(max_cnt_convex)

        # display results
        if self.verbose == 1:
            self.display_contours_restuls(_image, mask_cnt_all, mask_cnt, mask_covex, shape)
        else:
            print("{}: {}".format(self.image_path, shape))
        return None

    def color_based_segmentation(self, image):
        # convert image to different formats
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # create masks for colors using inRange method
        green_mask = cv2.inRange(hsv_frame, self.colors["low_green"], self.colors["high_green"])
        yellow_mask = cv2.inRange(hsv_frame, self.colors["low_yellow"], self.colors["high_yellow"])

        # performing bitwise operation to extract the region from the image containing specific color
        green = cv2.bitwise_and(frame, frame, mask=green_mask)
        yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)
        _width = frame.shape[0]
        _height = frame.shape[1]
        _frame_area = _width * _height

        # method to return the contours from the segmented region
        def _return_contours(_segment):
            _gray = cv2.cvtColor(_segment, cv2.COLOR_BGR2GRAY)
            _ret, _threshold = cv2.threshold(_gray, 80, 255, cv2.THRESH_BINARY)
            _contours, _hierarchy = cv2.findContours(_threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            return _contours

        # combine all contours found
        cnt_to_consider = []
        for mask in [yellow, green]:
            cnt_to_consider = [*cnt_to_consider,
                               *[cnt for cnt in _return_contours(mask) if cv2.contourArea(cnt) > _frame_area * 0.01]]

        # check if the shape of image tells the shape of the road sign :)
        if _width > _height and (_height / _width) < 0.7 or _width < _height and (_width / _height) < 0.7:
            # if so, then directly predict the box
            params = self.box_prediction(cnt_to_consider)
            if self.verbose == 1:
                self.display_box_results(image, cnt_to_consider, params)
            else:
                if len(params) > 0:
                    print("{}: {}".format(self.image_path, params[4]))

        # else we need to proceess the contours again for shape prediction
        else:
            self._countors_based_segmentation(frame, cnt_to_consider, flag=1)

    def _detect_shape_contours_approx(self, cnt):
        # finding the vertices of the contours using openCV buildin function
        _perimeter = cv2.arcLength(cnt, True)
        _vertices = cv2.approxPolyDP(cnt, 0.012 * _perimeter, True)
        shape = "unknown"

        # classifying the shape based on the number of vertices
        if len(_vertices) == 3:
            shape = "triangle"
        elif len(_vertices) == 4:
            # if shape has four vertices, then it can be either
            # 1. square
            # 2. Verticle Rectangle
            # 3. Horizontal Rectangle OR
            # 4. Diamond
            (x, y, w, h) = cv2.boundingRect(_vertices)
            if h != 0:
                _ratio = w / float(h)
                # a square have an aspect ratio ~ 1.0
                if 0.85 <= _ratio <= 1.15:
                    _v_diff_1 = abs(_vertices[0][0][1] - _vertices[1][0][1])
                    _v_diff_2 = abs(_vertices[0][0][1] - _vertices[3][0][1])
                    _thresh = h * 0.1
                    if _v_diff_1 > _thresh and _v_diff_2 > _thresh:
                        shape = "diamond"
                    else:
                        shape = "square"
                else:
                    shape = "horizontal rectangle" if w > h else "vertical rectangle"
        elif len(_vertices) == 5 or len(_vertices) == 6:
            shape = "pentagon"
        elif len(_vertices) == 7:
            shape = "heptagon"
        elif len(_vertices) == 8:
            shape = "octagon"
        else:
            shape = "circle"
        return shape

    def box_prediction(self, contours):
        boxes = []
        box_params = []
        # compute boxes for all contours
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        # then combining all small boxes the predict one large box
        if len(boxes) > 0:
            boxes = np.asarray(boxes)
            left, top = np.min(boxes, axis=0)[:2]
            right, bottom = np.max(boxes, axis=0)[2:]
            _box_width = abs(right - left)
            _box_height = abs(bottom - top)

            _ratio = _box_width / _box_height
            if 0.90 <= _ratio <= 1.10:
                shape = 'square'
            else:
                shape = 'vertical rectangle' if _box_width < _box_height else 'horizontal rectangle'
            box_params = [left, top, right, bottom, shape]
        return box_params

    def display_contours_restuls(self, image, cnts, extracted_cnts, processed_cnt, shape):
        # display all images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))
        fig.suptitle('Shape Detected: {}'.format(shape), fontsize=16)
        axes[0].imshow(image)
        axes[0].title.set_text("Quantized Image (KMeans)")
        axes[1].imshow(gray, cmap='gray')
        axes[1].title.set_text("Gray Scaled Image")
        axes[2].imshow(cnts, cmap='gray')
        axes[2].title.set_text("Large Area Contours")
        axes[3].imshow(extracted_cnts, cmap='gray')
        axes[3].title.set_text("Single Contour Extracted")
        axes[4].imshow(processed_cnt, cmap='gray')
        axes[4].title.set_text("Filled Holes")
        plt.show()

    def display_box_results(self, image, contours, params):
        # display all images
        if len(params) > 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            mask_contours = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask_contours, contours, -1, 255, thickness=1)

            mask_box = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask_box = cv2.rectangle(mask_box, (params[0], params[1]), (params[2], params[3]), 255, thickness=-1)

            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
            fig.suptitle('Shape Detected: {}'.format(params[4]), fontsize=16)
            axes[0].imshow(image)
            axes[0].title.set_text("Quantized Image (KMeans)")
            axes[1].imshow(gray, cmap='gray')
            axes[1].title.set_text("Gray Scaled Image")
            axes[2].imshow(mask_contours, cmap='gray')
            axes[2].title.set_text("Countors Detected")
            axes[3].imshow(mask_box, cmap='gray')
            axes[3].title.set_text("Bounding Box Prediction")
            plt.show()

    def preprcess_image(self, _image, n=8):
        # https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
        (h, w) = _image.shape[:2]
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2LAB)
        _image = _image.reshape((_image.shape[0] * _image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=n)
        labels = clt.fit_predict(_image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        _image = _image.reshape((h, w, 3))
        # convert from L*a*b* to RGB
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        return quant

    def predict_shape(self, image_path):
        self.image_path = image_path
        # read the image
        _image = cv2.imread(image_path)
        # preprocess the image
        _image_quantized = self.preprcess_image(_image)

        # convert image to grayscale and blur it using Gaussian blur
        _image_grey = cv2.cvtColor(_image_quantized, cv2.COLOR_BGR2GRAY)
        _image_grey = cv2.GaussianBlur(_image_grey, (3, 3), 0)
        # perform canny edge detection
        _edges = cv2.Canny(_image_grey, 50, 200, None, 3)

        # detect contours using a builtin function
        contours, hierarchy = cv2.findContours(_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnt_closed = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][2] != -1]
        # pass the contours for further processing
        self._countors_based_segmentation(_image, cnt_closed, flag=0)
