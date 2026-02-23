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

if not os.path.exists("images"):
    os.makedirs("images")

print("Press Space to capture 4 images. Press q to exit.")

captured = []
currentlyCaptured = 0

while currentlyCaptured < 4:
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    
    frame = cv2.resize(frame, (240, 240))
    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1)
    if key == ord(' '):
        currentlyCaptured += 1

        cv2.imwrite(f"images/{currentlyCaptured}.png", frame)
        captured.append(frame)
        
        print(f"Captured {currentlyCaptured}/4")
    elif key == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()

if currentlyCaptured == 4:
    h, w = 240, 240
    mosaic = numpy.zeros((h*2, w*2, 3), dtype=numpy.uint8)
    
    mosaic[0:h, 0:w] = captured[0]      
    mosaic[0:h, w:w*2] = captured[1]    
    mosaic[h:h*2, 0:w] = captured[2]    
    mosaic[h:h*2, w:w*2] = captured[3]  

    kernel = numpy.array([[ -1,  -1,  -1],
                      [ -1, 9,  -1],
                      [ -1,  -1,  -1]])
    mosaic[0:h, 0:w] = cv2.filter2D(mosaic[0:h, 0:w], -1, kernel, borderType=cv2.BORDER_CONSTANT)

    part2_source = mosaic[0:h, w:w*2].copy()
    for y in range(h):
        for x in range(w):
            mosaic[x, (w*2-1) - y] = part2_source[y, x]

    mosaic[h:h*2, 0:w, 0] = 0 
    mosaic[h:h*2, 0:w, 1] = 0

    print(f"\nType: {mosaic.dtype}")
    print(f"Shape: {mosaic.shape}")
    print(f"Size: {mosaic.size}")

    cv2.imshow("Final Mosaic", mosaic)
    cv2.imwrite("images/final_mosaic.png", mosaic)
    cv2.waitKey(0)

cv2.destroyAllWindows()