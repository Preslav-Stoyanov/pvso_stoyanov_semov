from ximea import xiapi
import cv2
import numpy as np
import os
import glob

# --- INITIALIZATION ---
cam = xiapi.Camera()
cam.open_device()

# Configure Camera
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB24")
cam.set_param("auto_wb", 1)

img = xiapi.Image()
cam.start_acquisition()

# Create directories
for folder in ["calibration_images", "results"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

chessboard_size = (7, 5)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0) ...
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane
captured = 0

# --- OPTIMIZED PHASE 1: CAPTURE ---
print("--- PHASE 1: CAPTURE (Optimized) ---")

# 1. Lower exposure to reduce motion blur
cam.set_exposure(20000)  # 20ms - better for moving objects

while True:
    cam.get_image(img)
    frame_rgb = img.get_image_data_numpy()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # 2. Performance Fix: Work on a smaller image for detection
    # This makes findChessboardCorners 4x-8x faster
    small_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale_factor = 0.5  # Scale down to 50%
    reduced = cv2.resize(small_gray, (0, 0), fx=scale_factor, fy=scale_factor)

    # Search for corners on the small image
    ret_live, corners_small = cv2.findChessboardCorners(reduced, chessboard_size,
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    display = cv2.resize(frame, (800, 600))

    if ret_live:
        # Scale corners back up for display/saving
        corners_live = corners_small / scale_factor
        # Draw on the 800x600 display (requires re-scaling corners for that specific window)
        display_scale = 800 / frame.shape[1]
        cv2.drawChessboardCorners(display, chessboard_size, corners_live * display_scale, ret_live)

    cv2.putText(display, f"Captured: {captured}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Live", display)

    key = cv2.waitKey(1)
    if key == ord(' '):
        if ret_live:
            captured += 1
            # SAVE THE FULL RESOLUTION IMAGE for better calibration math
            cv2.imwrite(f"calibration_images/chess_{captured}.png", frame)
            print(f"Captured {captured}")
        else:
            print("No corners detected! Keep the board steady and in focus.")

    elif key == ord('c'):
        break
# --- PHASE 2: CALIBRATION ---
cv2.destroyAllWindows()
images = glob.glob('calibration_images/*.png')

print("\nProcessing images for calibration...")
last_gray_shape = None

for fname in images:
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    last_gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
        cv2.imshow('Calibration Processing', image)
        cv2.waitKey(100)

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, last_gray_shape, None, None)
    print("\nCalibration successful!")

    # --- ADD THESE LINES TO MEET REQUIREMENTS ---
    print("\nIntrinsic Camera Matrix:\n", mtx)

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print(f"\nExtracted Parameters:")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")
    # --------------------------------------------

    np.save("results/camera_matrix.npy", mtx)
    np.save("results/dist_coeffs.npy", dist)

# Pre-calculate Undistortion Maps for speed
h, w = (480, 640)  # Target display size
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), 5)

# --- PHASE 3: DETECTION ---
print("\nStarting Real-time Detection...")
while True:
    cam.get_image(img)
    frame_rgb = img.get_image_data_numpy()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (w, h))

    # Fast Undistort
    undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # Red Color Masking
    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Highlight red objects in green (as per your original code)
    undistorted[mask > 0] = [0, 255, 0]

    # Shape Detection
    gray_det = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_det, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800: continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx_s, cy_s = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        num_corners = len(approx)
        shape = "Unknown"
        if num_corners == 3:
            shape = "Triangle"
        elif num_corners == 4:
            x, y, wc, hc = cv2.boundingRect(approx)
            ratio = wc / float(hc)
            shape = "Square" if 0.9 <= ratio <= 1.1 else "Rectangle"
        elif num_corners > 7:
            shape = "Circle"

        cv2.drawContours(undistorted, [approx], -1, (255, 0, 0), 2)
        cv2.putText(undistorted, shape, (cx_s - 20, cy_s), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Detection Result", undistorted)
    if cv2.waitKey(1) == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()