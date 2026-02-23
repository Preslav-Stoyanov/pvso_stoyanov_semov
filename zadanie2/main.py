from ximea import xiapi
import cv2
import numpy

# ---------------- CAMERA SETUP ----------------
cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB24")
cam.set_param("auto_wb", 1)

img = xiapi.Image()
cam.start_acquisition()

print("Press q to exit.")

# ---------------- MAIN LOOP ----------------
while True:
    cam.get_image(img)
    frame = img.get_image_data_numpy()

    # Resize for easier processing
    frame = cv2.resize(frame, (640, 480))

    output = frame.copy()

    # ---------------- PREPROCESSING ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ---------------- FIND CONTOURS ----------------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        area = cv2.contourArea(contour)
        if area < 500:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        shape = "Unknown"

        # ---------------- SHAPE CLASSIFICATION ----------------
        if len(approx) == 3:
            shape = "Triangle"

        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.9 <= aspect_ratio <= 1.1:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif len(approx) > 4:
            # Check circularity
            circularity = 4 * numpy.pi * area / (peri * peri)
            if circularity > 0.8:
                shape = "Circle"

        # ---------------- DRAW RESULTS ----------------
        if shape != "Unknown":
            cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)

            # Centroid calculation
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw centroid
                cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

                # Label shape
                cv2.putText(output, shape, (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 0), 2)

    # ---------------- DISPLAY ----------------
    cv2.imshow("Shape Detection", output)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# ---------------- CLEANUP ----------------
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()