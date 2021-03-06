import glob

import cv2
import numpy as np
from moviepy.editor import VideoFileClip


# Static functions for image transformation, color spaces changes etc.
def white_yellow_hls_mask(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask_yellow = cv2.inRange(img_hls, np.array([10, 0, 100]), np.array([40, 255, 255]))
    mask_white = cv2.inRange(img_hls, np.array([0, 210, 0]), np.array([255, 255, 255]))
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    return combined_mask


def s_bin_channel(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = img_hls[:, :, 2]
    thresh = (80, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def l_bin_channel(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = img_hls[:, :, 1]
    thresh = (180, 255)
    binary = np.zeros_like(L)
    binary[(L > thresh[0]) & (L <= thresh[1])] = 1
    return binary


def sobel(img, sobel_kernel=5, thresh=(0.7, 1.3)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)
    directed_grad = np.arctan2(sobely_abs, sobelx_abs)
    binary_output = np.zeros_like(directed_grad)
    binary_output[(directed_grad >= thresh[0]) & (directed_grad <= thresh[1])] = 1
    return binary_output


def masked_sobel(img, sobel_kernel=5, thresh=(0.7, 1.3)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)
    directed_grad = np.arctan2(sobely_abs, sobelx_abs)
    binary_output = np.zeros_like(directed_grad)
    binary_output[(directed_grad >= thresh[0]) & (directed_grad <= thresh[1])] = 1
    return binary_output


class LanesProcessor:
    def __init__(self, input_shape, kernel_size):
        self.input_shape = input_shape
        self.src_points = np.float32(
            [[568, 470], [722, 470], [218, 720], [1110, 720]])  # source points taken from straight lane image
        self.dst_points = np.float32([[300, 0], [950, 0], [300, 720], [950, 720]])  # points mapped to warp image
        self.ksize = kernel_size  # kernel size for Sobel
        self.last_leftx = []  # storing here points for last detected lanes left and right - in case whenever we won't
        self.last_lefty = []  # find points for lanes in current frame we'll use points from previous one
        self.last_rightx = []
        self.last_righty = []
        self.last_left_fit = []  # arrays for storing results for fitted polynomials for last x frames to calculate
        self.last_right_fit = []  # mean value of left and right line fit - used to avoid jitter between frames
        self.lanes_memory_size = 20  # number of stored polyfit's for mean value

        images = glob.glob('camera_cal/calibration*.jpg')  # loading images for camera calibration

        object_points = []
        image_points = []

        objp = np.zeros((9 * 6, 3), np.float32)
        objp[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)

        # Camera calibration using images from camera_cal directory
        for filename in images:
            image = cv2.imread(filename)

            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image_ret, corners = cv2.findChessboardCorners(grayed, (9, 6))

            if image_ret:
                object_points.append(objp)
                image_points.append(corners)

        # Calibration of camera, setting calibration matrix and other calibration properties
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, input_shape, None, None)
        self.left_fit = []
        self.right_fit = []
        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

    def perspective_transform_mtx(self):
        return cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def perspective_transform_mtxinv(self):
        return cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def warp(self, img):
        return cv2.warpPerspective(img, self.perspective_transform_mtx(), (self.input_shape[0], self.input_shape[1]),
                                   flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.perspective_transform_mtxinv(), (self.input_shape[0], self.input_shape[1]),
                                   flags=cv2.INTER_LINEAR)

    #  Function for finding indices for left and right lanes for given binary warped image using
    #  traditional sliding window technique
    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 45

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(left_lane_inds) < minpix and good_left_inds.any():
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(right_lane_inds) < minpix and good_right_inds.any():
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        # If there's no pixels found for left or right lane we use points from previous frame
        leftx = nonzerox[left_lane_inds] if left_lane_inds.any() else self.last_leftx
        lefty = nonzeroy[left_lane_inds] if left_lane_inds.any() else self.last_lefty
        rightx = nonzerox[right_lane_inds] if right_lane_inds.any() else self.last_rightx
        righty = nonzeroy[right_lane_inds] if right_lane_inds.any() else self.last_righty
        # Store new indices for next frame
        self.last_leftx = leftx
        self.last_lefty = lefty
        self.last_rightx = rightx
        self.last_righty = righty

        return leftx, lefty, rightx, righty

    #  Function for finding indices for left and right lanes for given binary warped image using
    #  search within margin of previous polyfit
    def search_around_poly(self, binary_warped):
        # Width of the margin around the previous polynomial to search
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy +
                                       self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0] * (nonzeroy ** 2) +
                                                                                  self.left_fit[1] * nonzeroy +
                                                                                  self.left_fit[
                                                                                      2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy +
                                        self.right_fit[2] - margin)) & (
                                   nonzerox < (self.right_fit[0] * (nonzeroy ** 2) +
                                               self.right_fit[1] * nonzeroy + self.right_fit[
                                                   2] + margin)))

        # Extract left and right line pixel positions
        # If there's no pixels found for left or right lane we use points from previous frame
        leftx = nonzerox[left_lane_inds] if left_lane_inds.any() else self.last_leftx
        lefty = nonzeroy[left_lane_inds] if left_lane_inds.any() else self.last_lefty
        rightx = nonzerox[right_lane_inds] if right_lane_inds.any() else self.last_rightx
        righty = nonzeroy[right_lane_inds] if right_lane_inds.any() else self.last_righty
        # Store new indices for next frame
        self.last_leftx = leftx
        self.last_lefty = lefty
        self.last_rightx = rightx
        self.last_righty = righty

        return leftx, lefty, rightx, righty

    def fit_polynomial(self, binary_warped):
        # Find lane pixels using sliding window technique at first and then use searching withing last polyfit margin
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped) \
            # if (len(self.left_fit) is 0 and len(self.right_fit) is 0) else self.search_around_poly(binary_warped)

        # Fit polynomial to found points for left and right lane
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Add calculated polyfit to array for calculating mean polyfit value for last frames to smooth our lanes
        # between frames
        self.last_left_fit.append(self.left_fit)
        self.last_right_fit.append(self.right_fit)
        # If array size exceeds threshold for stored polyfits delete oldest calculation
        if len(self.last_left_fit) > self.lanes_memory_size:
            del self.last_left_fit[0]
        if len(self.last_right_fit) > self.lanes_memory_size:
            del self.last_right_fit[0]
        # Calculate mean left and right line for lanes
        mean_left_fit = np.mean(self.last_left_fit, axis=0)
        mean_right_fit = np.mean(self.last_right_fit, axis=0)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fitx = mean_left_fit[0] * ploty ** 2 + mean_left_fit[1] * ploty + mean_left_fit[2]
            right_fitx = mean_right_fit[0] * ploty ** 2 + mean_right_fit[1] * ploty + mean_right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        binary_warped = np.dstack((binary_warped, binary_warped, binary_warped))

        #  Prepare points for drawing area between lines
        leftline = np.flip(np.dstack((left_fitx, ploty)), 1)
        rightline = np.dstack((right_fitx, ploty))
        lines = np.concatenate((leftline, rightline), axis=1)
        #  Draw green area
        cv2.fillPoly(binary_warped, np.int32(lines), (0, 255, 0))
        return binary_warped

    def measure_curvature(self, img):
        # Define conversions in y from pixels space to meters
        ym_per_pix = 30 / self.input_shape[1]  # meters per pixel in y dimension
        mean_left_fit = np.mean(self.last_left_fit, axis=0)
        mean_right_fit = np.mean(self.last_right_fit, axis=0)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_eval = np.max(ploty)

        # Calculation for curvature in meters for left line
        left_curverad = np.power(1 + np.power(2 * mean_left_fit[0] * y_eval * ym_per_pix + mean_left_fit[1], 2),
                                 1.5) / np.absolute(
            mean_left_fit[0] * 2)
        # Calculation for curvature in meters for right line
        right_curverad = np.power(1 + np.power(2 * mean_right_fit[0] * y_eval * ym_per_pix + mean_right_fit[1], 2),
                                  1.5) / np.absolute(
            mean_right_fit[0] * 2)

        # Put calculated curvature to image - took mean value for left and right curvature
        curvature_in_meters = 'Curvature {0:6.2f} m'.format((left_curverad + right_curverad) / 2)
        cv2.putText(img, curvature_in_meters, (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=1,
                    lineType=2)
        return img

    def offset_from_lane_centre(self, img):
        # We warp lane left lane to 300px x position and for right at 950px which gives us 650px of lane width on image
        xm_per_pix = 3.7 / 650  # meters per pixel in x dimension
        mean_left_fit = np.mean(self.last_left_fit, axis=0)
        mean_right_fit = np.mean(self.last_right_fit, axis=0)
        # Calculating lane center on image
        lane_center = self.input_shape[0] / 2
        # Calculating position for left lane
        left_lane_base = mean_left_fit[0] * self.input_shape[1] ** 2 + mean_left_fit[1] * self.input_shape[1] + \
                         mean_left_fit[2]
        # Calculating position for right lane
        right_lane_base = mean_right_fit[0] * self.input_shape[1] ** 2 + mean_right_fit[1] * self.input_shape[1] + \
                          mean_right_fit[2]
        # Calculating lane center for current frame
        current_lane_center = (left_lane_base + right_lane_base) / 2
        # Calculate difference between current center and desired center
        offset = current_lane_center - lane_center
        # Put offset text to image
        offset_text = 'Distance from lane centre {0:3.2f}m to {1}'.format(abs(offset) * xm_per_pix,
                                                                          'right' if offset < 0 else 'left')
        cv2.putText(img, offset_text, (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=1,
                    lineType=2)
        return img

    # Whole pipeline for frame
    def process_image(self, img):
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        masked = white_yellow_hls_mask(undistorted)
        warped_img = self.warp(masked)
        poly = self.fit_polynomial(warped_img)
        unwarped = self.unwarp(poly)
        with_area = cv2.addWeighted(img, 1., unwarped, 0.3, 0.)
        with_curvature = self.measure_curvature(with_area)
        with_offset = self.offset_from_lane_centre(with_curvature)
        return with_offset

    # Process video and save transformation result to output.mp4 file
    def process_video(self, vid_path):
        result = 'output_images/output.mp4'
        clip1 = VideoFileClip(vid_path)
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(result, audio=False)
