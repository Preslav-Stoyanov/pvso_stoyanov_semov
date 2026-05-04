from ximea import xiapi
import cv2
import os
import time

# =============================================================================
# NASTAVENIA - nastav pred snímaním a NEMEŇ počas celého snímania!
# =============================================================================

EXPOSURE_US   = 10000   # expozícia v µs – uprav podľa svetla (vyššie = jasnejší)
GAIN_DB       = 0.0     # gain (nechaj 0, zvyšuj iba ak je obraz tmavý)

AUTO_EXPOSURE = False   # MUSÍ byť False – fixné nastavenia počas celého snímania
AUTO_WB       = False   # MUSÍ byť False

WB_RED        = 1.7
WB_GREEN      = 1.0
WB_BLUE       = 1.3

IMAGE_FORMAT  = "XI_RGB24"

OUTPUT_DIR    = "captured_images"   # priečinok kde sa uložia fotky

# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
shot_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")])

cam = xiapi.Camera()
print("Otváranie kamery...")
cam.open_device()
print(f"Kamera: {cam.get_device_name()}  |  SN: {cam.get_device_sn()}")

cam.disable_aeag()
cam.set_exposure(EXPOSURE_US)
cam.set_gain(GAIN_DB)
print(f"Expozícia: {cam.get_exposure()} µs  |  Gain: {cam.get_gain()} dB")

cam.disable_auto_wb()
cam.set_wb_kr(WB_RED)
cam.set_wb_kg(WB_GREEN)
cam.set_wb_kb(WB_BLUE)

cam.set_imgdataformat(IMAGE_FORMAT)

img = xiapi.Image()
cam.start_acquisition()

print("=" * 50)
print(f"Uložené fotky: {OUTPUT_DIR}/")
print("SPACE / ENTER = uložiť fotku")
print("Q             = ukončiť")
print("=" * 50)

while True:
    cam.get_image(img)
    frame = img.get_image_data_numpy()

    # Zmenšená náhľad pre zobrazenie (pôvodná veľkosť sa uloží)
    preview = cv2.resize(frame, (1280, 720))

    # Overlay s počtom fotografií
    cv2.putText(preview, f"Fotky: {shot_count}  |  SPACE=foto  Q=koniec",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Farebný indikátor – zelený = pripravený, červený = práve uložil
    cv2.imshow("XIMEA kamera", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord(" "), 13):   # SPACE alebo ENTER
        filename = os.path.join(OUTPUT_DIR, f"IMG_{shot_count:04d}.jpg")
        # Uložiť v plnom rozlíšení
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shot_count += 1
        print(f"[{shot_count:>3}] Uložené: {filename}")

        # Krátky vizuálny feedback – biely flash
        flash = preview.copy()
        flash[:] = (255, 255, 255)
        cv2.imshow("XIMEA kamera", flash)
        cv2.waitKey(80)

    elif key == ord("q"):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
print(f"\nHotovo. Celkom uložených fotiek: {shot_count}")
print(f"Fotky sú v priečinku: {os.path.abspath(OUTPUT_DIR)}/")