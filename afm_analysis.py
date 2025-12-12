import numpy as np
import pandas as pd
import cv2

def analyze_triangle_holes(img_bgr, field_um: float, min_area_px: int = 50, blur_sigma: float = 1.0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    um_per_px = field_um / float(w)

    # 平滑化
    k = int(max(3, (blur_sigma * 6) // 2 * 2 + 1))  # odd
    blur = cv2.GaussianBlur(gray, (k, k), blur_sigma)

    # Otsu 2値化（黒い穴を白に）
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # 輪郭
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rows = []
    centers = []
    kept = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 正三角形：A = sqrt(3)/4 L^2
        L_px = np.sqrt(4.0 * area / np.sqrt(3.0))
        L_um = L_px * um_per_px

        rows.append({
            "area_px2": area,
            "area_um2": area * (um_per_px ** 2),
            "L_um": L_um,
            "cx_um": cx * um_per_px,
            "cy_um": cy * um_per_px,
        })
        centers.append([cx, cy])
        kept.append(c)

    if len(rows) == 0:
        return None

    df = pd.DataFrame(rows)

    # 周期：重心の最近接距離（代表値として中央値を採用）
    centers = np.array(centers, dtype=float)
    nn = []
    for i in range(len(centers)):
        d = np.linalg.norm(centers - centers[i], axis=1)
        d = d[d > 1e-9]
        nn.append(np.min(d))
    pitch_px = float(np.median(nn))
    pitch_um = pitch_px * um_per_px

    summary = {
        "n": len(df),
        "L_mean_um": float(df["L_um"].mean()),
        "L_std_um": float(df["L_um"].std(ddof=1)) if len(df) > 1 else 0.0,
        "pitch_um": pitch_um,
        "um_per_px": um_per_px,
    }

    # 可視化
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, kept, -1, (0, 0, 255), 1)
    for (cx, cy) in centers:
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 2, (0, 255, 0), -1)

    # Streamlit表示用にRGBへ
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return summary, df, bin_img, overlay_rgb
