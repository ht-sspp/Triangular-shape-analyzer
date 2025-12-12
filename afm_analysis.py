import numpy as np
import pandas as pd
import cv2

def analyze_triangle_holes(
    img_bgr,
    um_per_px: float,
    min_area_px: int = 50,
    blur_sigma: float = 1.0,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

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
    kept_contours = []
    centers_px = []

    # 先に面積フィルタして残す（idは0..で振り直す）
    tmp = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        tmp.append((c, area, cx, cy))

    if len(tmp) == 0:
        return None

    # id付与（上から順でもOKだし、x/yでソートしてもOK）
    # ここでは「左上→右下」に並ぶようにソートして番号を振ります
    tmp.sort(key=lambda t: (t[3], t[2]))  # (cy, cx)

    for hid, (c, area, cx, cy) in enumerate(tmp):
        # 正三角形：A = sqrt(3)/4 L^2
        L_px = np.sqrt(4.0 * area / np.sqrt(3.0))
        L_um = L_px * um_per_px

        x, y, bw, bh = cv2.boundingRect(c)

        rows.append({
            "id": hid,
            "L_um": L_um,
            "area_um2": area * (um_per_px ** 2),
            "cx_um": cx * um_per_px,
            "cy_um": cy * um_per_px,
            "cx_px": cx,
            "cy_px": cy,
            "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh,
        })
        kept_contours.append(c)
        centers_px.append((cx, cy))

    df = pd.DataFrame(rows)

    # 周期：最近接距離の中央値
    centers = np.array(centers_px, dtype=float)
    nn = []
    for i in range(len(centers)):
        d = np.linalg.norm(centers - centers[i], axis=1)
        d = d[d > 1e-9]
        nn.append(np.min(d))
    pitch_px = float(np.median(nn))
    pitch_um = pitch_px * um_per_px

    summary = {
        "n": int(len(df)),
        "L_mean_um": float(df["L_um"].mean()),
        "L_std_um": float(df["L_um"].std(ddof=1)) if len(df) > 1 else 0.0,
        "pitch_um": float(pitch_um),
        "um_per_px": float(um_per_px),
        "field_um": float(um_per_px * w),
    }

    # 可視化：輪郭＋重心＋ID文字
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, kept_contours, -1, (0, 0, 255), 1)
    for r in rows:
        cx = int(round(r["cx_px"]))
        cy = int(round(r["cy_px"]))
        cv2.circle(overlay, (cx, cy), 2, (0, 255, 0), -1)
        cv2.putText(
            overlay, str(r["id"]), (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA
        )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return summary, df, bin_img, overlay_rgb, kept_contours, centers_px
