from ximea import xiapi
import cv2
import numpy
import os

cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB24")
cam.set_param("auto_wb", 1)

img = xiapi.Image()
cam.start_acquisition()

for folder in ["calibration_images", "results"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

chessboard_size = (7, 5) 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = numpy.zeros((chessboard_size[0] * chessboard_size[1], 3), numpy.float32)
objp[:, :2] = numpy.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 
captured = 0

print("Press SPACE to capture.")
print("Press 'C' to start calibration.")

while True:
    cam.get_image(img)
    frame_rgb = img.get_image_data_numpy()
    
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_chess, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    display = frame.copy()
    if ret_chess:
        cv2.drawChessboardCorners(display, chessboard_size, corners, ret_chess)

    cv2.putText(display, f"Captured: {captured}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Calibration Capture", display)

    key = cv2.waitKey(1)
    if key == ord(' '):
        if ret_chess:
            captured += 1
            cv2.imwrite(f"calibration_images/chess_{captured}.png", frame)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"Captured image {captured}")
    elif key == ord('c') and captured > 0:
        break

print("\nComputing Calibration...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\nIntrinsic Camera Matrix:\n", mtx)
print(f"fx: {mtx[0,0]}, fy: {mtx[1,1]}, cx: {mtx[0,2]}, cy: {mtx[1,2]}")

numpy.save("results/camera_matrix.npy", mtx)
numpy.save("results/dist_coeffs.npy", dist)

print("\nStarting Real-time Task... Press 'Q' to quit.")
while True:
    cam.get_image(img)
    frame_rgb = img.get_image_data_numpy()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    undistorted = cv2.undistort(frame, mtx, dist)

    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
    lower_red = numpy.array([0, 120, 70])
    upper_red = numpy.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    result_frame = undistorted.copy()
    result_frame[mask > 0] = [0, 255, 0] 

    gray_det = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    circle_blur = cv2.medianBlur(gray_det, 5)
    circles = cv2.HoughCircles(circle_blur, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))
        for i in circles[0, :]:
            cv2.circle(result_frame, (i[0], i[1]), i[2], (255, 0, 0), 3)
            cv2.circle(result_frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.putText(result_frame, "Circle", (i[0]-20, i[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    edges = cv2.Canny(cv2.GaussianBlur(gray_det, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 1500: continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        
        num_v = len(approx)
        
        if num_v == 3 or num_v == 4:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                label = "Triangle" if num_v == 3 else "Square/Rect"
                
                cv2.drawContours(result_frame, [approx], -1, (255, 0, 0), 3)
                cv2.circle(result_frame, (cX, cY), 5, (255, 255, 255), -1)
                cv2.putText(result_frame, label, (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Final Result", result_frame)
    if cv2.waitKey(1) == ord('q'): break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()