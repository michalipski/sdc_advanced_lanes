import glob
import sys

import cv2
import numpy as np

from adv_lanes_processor import LanesProcessor

images_shape = (1280, 720)
src_points = np.float32([[568, 470], [704, 470], [231, 720], [1092, 720]])
dst_points = np.float32([[300, 0], [950, 0], [300, 720], [950, 720]])
M = cv2.getPerspectiveTransform(src_points, dst_points)
Minv = cv2.getPerspectiveTransform(dst_points, src_points)
ksize = 3  # kernel size for Sobel


def calibrate_camera():
    images = glob.glob('camera_cal/calibration*.jpg')

    object_points = []
    image_points = []

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)

    for filename in images:
        image = cv2.imread(filename)

        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_ret, corners = cv2.findChessboardCorners(grayed, (9, 6))

        if image_ret:
            object_points.append(objp)
            image_points.append(corners)

    return cv2.calibrateCamera(object_points, image_points, images_shape, None, None)


def white_yellow_hls_mask(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask_yellow = cv2.inRange(img_hls, np.array([10, 0, 100]), np.array([40, 255, 255]))
    mask_white = cv2.inRange(img_hls, np.array([0, 210, 0]), np.array([255, 255, 255]))
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    return combined_mask


def s_bin_channel(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = img_hls[:, :, 2]
    thresh = (23, 164)
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

def combined_channels(img):
    binary_s = s_bin_channel(img)
    binary_l = l_bin_channel(img)
    return cv2.bitwise_or(binary_s, binary_l)


def warped(img):
    return cv2.warpPerspective(img, M, (images_shape[0], images_shape[1]), flags=cv2.INTER_LINEAR)


def unwarped(img):
    return cv2.warpPerspective(img, Minv, (images_shape[0], images_shape[1]), flags=cv2.INTER_LINEAR)


def sobel(img, sobel_kernel=7, thresh=(0.7, 1.3)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)
    directed_grad = np.arctan2(sobely_abs, sobelx_abs)
    binary_output = np.zeros_like(directed_grad)
    binary_output[(directed_grad >= thresh[0]) & (directed_grad <= thresh[1])] = 1
    return binary_output


ret, mtx, dist, rvecs, tvecs = calibrate_camera()


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

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
    leftx = nonzerox[left_lane_inds] if left_lane_inds.any() else last_leftx
    lefty = nonzeroy[left_lane_inds] if left_lane_inds.any() else last_lefty
    rightx = nonzerox[right_lane_inds] if right_lane_inds.any() else last_rightx
    righty = nonzeroy[right_lane_inds] if right_lane_inds.any() else last_righty

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.fill_betweenx(ploty, left_fitx, right_fitx, facecolor='green', alpha=0.6)
    leftline = np.flip(np.dstack((left_fitx, ploty)), 1)
    rightline = np.dstack((right_fitx, ploty))
    lines = np.concatenate((leftline, rightline), axis=1)
    cv2.fillPoly(out_img, np.int32(lines), (0, 255, 0))
    return out_img


def process_image(img):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    masked = white_yellow_hls_mask(undistorted)
    warped_img = warped(masked)
    poly = fit_polynomial(warped_img)
    unwarp = unwarped(poly)
    final_image = cv2.addWeighted(img, 0.7, unwarp, 1., 0.)
    return final_image


lanes_processor = LanesProcessor(images_shape, ksize)
lanes_processor.process_video(sys.argv[1])
