from ximea import xiapi
import cv2
import numpy
import os
import glob


cam = xiapi.Camera()
cam.open_device()

cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB24")
cam.set_param("auto_wb", 1)

img = xiapi.Image()
cam.start_acquisition()


if not os.path.exists("calibration_images"):
    os.makedirs("calibration_images")

if not os.path.exists("results"):
    os.makedirs("results")

print("Press SPACE to capture calibration images")
print("Press C to start calibration")
print("Press Q to quit")



chessboard_size = (7,6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = numpy.zeros((chessboard_size[0]*chessboard_size[1],3), numpy.float32)
objp[:,:2] = numpy.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

objpoints = []
imgpoints = []

captured = 0


while True:

    cam.get_image(img)
    frame = img.get_image_data_numpy()

    frame = cv2.resize(frame,(640,480))
    display = frame.copy()

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1)

    if key == ord(' '):

        captured += 1

        filename = f"calibration_images/chess_{captured}.png"
        cv2.imwrite(filename, frame)

        print("Saved:", filename)

    elif key == ord('c'):
        break

    elif key == ord('q'):
        cam.stop_acquisition()
        cam.close_device()
        cv2.destroyAllWindows()
        exit()


images = glob.glob('calibration_images/*.png')

for fname in images:

    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret == True:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(image,chessboard_size,corners2,ret)

        cv2.imshow('Corners', image)
        cv2.waitKey(200)

cv2.destroyAllWindows()

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\nCamera Matrix:\n", cameraMatrix)

fx = cameraMatrix[0,0]
fy = cameraMatrix[1,1]
cx = cameraMatrix[0,2]
cy = cameraMatrix[1,2]

print("\nfx =", fx)
print("fy =", fy)
print("cx =", cx)
print("cy =", cy)

numpy.save("results/camera_matrix.npy", cameraMatrix)
numpy.save("results/dist_coeffs.npy", distCoeffs)

print("\nCalibration saved.")


print("\nRunning real-time detection")

while True:

    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv2.resize(frame,(640,480))



    h, w = frame.shape[:2]

    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix,
        distCoeffs,
        (w,h),
        1,
        (w,h)
    )

    undistorted = cv2.undistort(
        frame,
        cameraMatrix,
        distCoeffs,
        None,
        newCameraMatrix
    )


    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)

    lower_red1 = numpy.array([0,120,70])
    upper_red1 = numpy.array([10,255,255])

    lower_red2 = numpy.array([170,120,70])
    upper_red2 = numpy.array([180,255,255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    undistorted[mask > 0] = [0,255,0]



    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,50,150)

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 1000:
            continue

        approx = cv2.approxPolyDP(
            cnt,
            0.02*cv2.arcLength(cnt,True),
            True
        )

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx_shape = int(M["m10"]/M["m00"])
        cy_shape = int(M["m01"]/M["m00"])

        shape = "Unknown"

        if len(approx) == 3:
            shape = "Triangle"

        elif len(approx) == 4:

            x,y,wc,hc = cv2.boundingRect(approx)

            ratio = wc/float(hc)

            if ratio > 0.95 and ratio < 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif len(approx) > 6:
            shape = "Circle"

        cv2.drawContours(undistorted,[approx],-1,(0,255,0),3)

        cv2.circle(undistorted,(cx_shape,cy_shape),5,(255,0,0),-1)

        cv2.putText(
            undistorted,
            shape,
            (cx_shape-40,cy_shape-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )


    cv2.imshow("Undistorted + Detection", undistorted)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()