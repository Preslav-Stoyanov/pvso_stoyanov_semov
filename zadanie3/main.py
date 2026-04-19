import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_manual_hist(channel):
    hist = np.zeros(256, dtype=np.int64)
    flat = channel.flatten()
    for intensity in flat:
        hist[intensity] += 1
    return hist

def manual_equalize(gray_img):
    hist = get_manual_hist(gray_img)
    cdf  = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    lut = np.round(
        (cdf - cdf_min) / (gray_img.size - cdf_min) * 255
    ).clip(0, 255).astype('uint8')
    return lut[gray_img]

def manual_clahe_simple(image, grid=(8, 8), clip_limit=2.0):
    h, w   = image.shape
    rows, cols = grid

    pad_h = (rows - h % rows) % rows
    pad_w = (cols - w % cols) % cols
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    th, tw = ph // rows, pw // cols

    luts = np.zeros((rows, cols, 256), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            tile  = padded[i*th:(i+1)*th, j*tw:(j+1)*tw]
            hist  = get_manual_hist(tile).astype(np.float32)
            limit  = clip_limit * tile.size / 256.0
            excess = np.sum(np.maximum(hist - limit, 0))
            hist   = np.minimum(hist, limit) + excess / 256.0
            cdf     = hist.cumsum()
            cdf_min = cdf[cdf > 0][0]
            lut     = (cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-6) * 255.0
            luts[i, j] = np.clip(lut, 0, 255)

    cy = (np.arange(rows) + 0.5) * th
    cx = (np.arange(cols) + 0.5) * tw

    ys = np.arange(ph, dtype=np.float32)
    xs = np.arange(pw, dtype=np.float32)

    r_hi = np.searchsorted(cy, ys, side='right').clip(0, rows - 1)
    r_lo = (r_hi - 1).clip(0, rows - 1)
    wy = np.where(cy[r_hi] > cy[r_lo], (ys - cy[r_lo]) / (cy[r_hi] - cy[r_lo]), 0.0).clip(0, 1)

    c_hi = np.searchsorted(cx, xs, side='right').clip(0, cols - 1)
    c_lo = (c_hi - 1).clip(0, cols - 1)
    wx = np.where(cx[c_hi] > cx[c_lo], (xs - cx[c_lo]) / (cx[c_hi] - cx[c_lo]), 0.0).clip(0, 1)

    WY = wy[:, None];  WX = wx[None, :]
    RL = r_lo[:, None].astype(int);  RH = r_hi[:, None].astype(int)
    CL = c_lo[None, :].astype(int);  CH = c_hi[None, :].astype(int)
    P  = padded.astype(int)

    out = (
        (1 - WY) * (1 - WX) * luts[RL, CL, P] +
        (1 - WY) *      WX  * luts[RL, CH, P] +
             WY  * (1 - WX) * luts[RH, CL, P] +
             WY  *      WX  * luts[RH, CH, P]
    )
    return np.clip(out[:h, :w], 0, 255).astype(np.uint8)

_fig_gray = None
_fig_rgb  = None

def plot_histograms(original, m_eq, cv_eq, m_clahe, cv_clahe, title_prefix=""):
    global _fig_gray
    if _fig_gray is None or not plt.fignum_exists(_fig_gray.number):
        _fig_gray, axes = plt.subplots(1, 5, figsize=(18, 3), num="Grayscale Histograms")
        plt.show(block=False)
    else:
        _fig_gray.clf()
        axes = _fig_gray.subplots(1, 5)

    _fig_gray.suptitle(f"{title_prefix}Grayscale Histograms", fontsize=12)
    labels = ["Original", "Manual Eq", "OpenCV Eq", "Manual CLAHE", "OpenCV CLAHE"]
    for ax, img, label in zip(axes, [original, m_eq, cv_eq, m_clahe, cv_clahe], labels):
        ax.plot(get_manual_hist(img), color='gray')
        ax.set_title(label, fontsize=9)
        ax.set_xlim([0, 255])
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")

    _fig_gray.tight_layout()
    _fig_gray.canvas.draw()
    _fig_gray.canvas.flush_events()

def plot_rgb_histogram(frame_bgr, title_prefix=""):
    global _fig_rgb
    if _fig_rgb is None or not plt.fignum_exists(_fig_rgb.number):
        _fig_rgb, ax = plt.subplots(figsize=(7, 3), num="RGB Histogram")
        plt.show(block=False)
    else:
        _fig_rgb.clf()
        ax = _fig_rgb.add_subplot(1, 1, 1)

    _fig_rgb.suptitle(f"{title_prefix}Manual RGB Histogram", fontsize=12)
    for name, idx, c in [('Blue', 0, 'blue'), ('Green', 1, 'green'), ('Red', 2, 'red')]:
        ax.plot(get_manual_hist(frame_bgr[:, :, idx]), color=c, label=name, alpha=0.8)

    ax.set_xlim([0, 255])
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.legend()
    _fig_rgb.tight_layout()
    _fig_rgb.canvas.draw()
    _fig_rgb.canvas.flush_events()

def process_frame(frame):
    small = cv2.resize(frame, (320, 240))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    m_eq    = manual_equalize(gray)
    m_clahe = manual_clahe_simple(gray)
    cv_eq   = cv2.equalizeHist(gray)
    cv_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def to_bgr(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    panels = [to_bgr(gray), to_bgr(m_eq), to_bgr(cv_eq),
              to_bgr(m_clahe), to_bgr(cv_clahe), np.zeros((240, 320, 3), dtype=np.uint8)]

    top_row  = np.hstack(panels[:3])
    bot_row  = np.hstack(panels[3:])
    combined = np.vstack((top_row, bot_row))

    font, scale, thick, colour = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (255, 255, 255)
    labels_top = ["1. Original", "2. Manual Eq", "3. OpenCV Eq"]
    labels_bot = ["4. Manual CLAHE", "5. OpenCV CLAHE", ""]
    for k, lbl in enumerate(labels_top):
        cv2.putText(combined, lbl, (k*320 + 8, 22), font, scale, colour, thick)
    for k, lbl in enumerate(labels_bot):
        cv2.putText(combined, lbl, (k*320 + 8, 262), font, scale, colour, thick)

    return combined, gray, m_eq, cv_eq, m_clahe, cv_clahe, small

if os.path.exists("images/test.png"):
    print(f"Testing file implementation...")
    file_img = cv2.imread("images/test.png")
    result, gray, m_eq, cv_eq, m_clahe, cv_clahe, small = process_frame(file_img)

    cv2.imshow("Test file", result)
    plot_histograms(gray, m_eq, cv_eq, m_clahe, cv_clahe, title_prefix="[File] ")
    plot_rgb_histogram(small, title_prefix="[File] ")

    while True:
        key = cv2.waitKey(50)
        _fig_gray.canvas.flush_events()
        _fig_rgb.canvas.flush_events()
        if key != -1:
            break

    cv2.destroyWindow("Test file")
    _fig_gray = _fig_rgb = None
    plt.close('all')

cap = cv2.VideoCapture(0)

print("Running — Q = quit   H = histograms")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display, gray, m_eq, cv_eq, m_clahe, cv_clahe, small = process_frame(frame)
    cv2.imshow("Group D: Histogram Enhancement", display)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('h'):
        plot_histograms(gray, m_eq, cv_eq, m_clahe, cv_clahe)
        plot_rgb_histogram(small)

cap.release()
cv2.destroyAllWindows()
plt.close('all')