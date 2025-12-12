import streamlit as st
from PIL import Image
import numpy as np
import cv2

from streamlit_cropper import st_cropper
from afm_analysis import analyze_triangle_holes

st.set_page_config(page_title="AFMメタサーフェス自動解析", layout="wide")
st.title("AFMメタサーフェス（三角穴）自動解析")

st.write("1) 画像をアップロード → 2) パターン領域だけドラッグで切り出し → 3) 切り出し領域の実サイズ(µm)を入力 → 4) 解析")

uploaded = st.file_uploader("AFM画像（png/jpg/tif/bmp）", type=["png", "jpg", "jpeg", "tif", "tiff","bmp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.sidebar.header("解析設定")
    scale_mode = st.sidebar.radio(
    "スケール指定方法",
    ["切り出し領域の一辺 [µm] を入力", "1ピクセルあたりの長さ [µm/px] を入力"]
    )

    field_um_input = None
    um_per_px_input = None

    if scale_mode == "切り出し領域の一辺 [µm] を入力":
        field_um_input = st.sidebar.number_input("切り出し領域の一辺 [µm]", min_value=0.01, value=5.0, step=0.1)
    else:
        um_per_px_input = st.sidebar.number_input("1 px あたりの長さ [µm/px]", min_value=1e-6, value=0.01, step=0.001, format="%.6f")
        field_um = st.sidebar.number_input(
            "切り出し領域の一辺の長さ [µm]（正方形スキャン想定）",
            min_value=0.01, value=5.0, step=0.1
        )
    min_area_px = st.sidebar.number_input("最小穴面積しきい値 [px^2]（ノイズ除去）", min_value=0, value=50, step=10)
    blur_sigma = st.sidebar.number_input("平滑化 sigma", min_value=0.0, value=1.0, step=0.5)

    crop_w_px, crop_h_px = cropped.size  # PILは (W,H)
    if field_um_input is not None:
        um_per_px = float(field_um_input) / float(crop_w_px)
    else:
        um_per_px = float(um_per_px_input)

    result = analyze_triangle_holes(
    cropped_bgr,
    um_per_px=um_per_px,
    min_area_px=int(min_area_px),
    blur_sigma=float(blur_sigma),
    )
    
    st.subheader("① 解析したい領域を切り出し（ファイル情報の表示部分を除外）")
    cropped = st_cropper(img, realtime_update=True, aspect_ratio=(1, 1))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="元画像", use_container_width=True)
    with col2:
        st.image(cropped, caption="切り出し後（解析対象）", use_container_width=True)

    if st.button("② 解析する"):
        # PIL -> OpenCV (BGR)
        cropped_np = np.array(cropped)
        cropped_bgr = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)

        result = analyze_triangle_holes(
            cropped_bgr,
            field_um=float(field_um),
            min_area_px=int(min_area_px),
            blur_sigma=float(blur_sigma),
        )

        if result is None:
            st.error("穴が検出できませんでした。min_area / sigma を調整するか、切り出し範囲を見直してください。")
        else:
            summary, df, bin_img, overlay_rgb, kept_contours, centers_px = result

            st.subheader("検出結果（番号＝表の id）")
            st.image(overlay_rgb, use_container_width=True)
            
            st.subheader("穴ごとの結果（idで画像と対応）")
            st.dataframe(df[["id","L_um","area_um2","cx_um","cy_um"]], use_container_width=True)
            
            # ---- 選んだ穴を強調表示＆拡大表示 ----
            st.subheader("特定の穴を選んで確認")
            hole_id = st.selectbox("hole id", df["id"].tolist(), index=0)
            
            # 強調オーバーレイ
            highlight = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR).copy()
            # 全輪郭は薄く
            cv2.drawContours(highlight, kept_contours, -1, (80, 80, 80), 1)
            # 選択穴は太く
            c_sel = kept_contours[int(hole_id)]
            cv2.drawContours(highlight, [c_sel], -1, (0, 0, 255), 3)
            
            # 選択穴のbbox周りを拡大表示
            row = df[df["id"] == hole_id].iloc[0]
            x, y, bw, bh = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
            pad = 20
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(highlight.shape[1], x + bw + pad); y1 = min(highlight.shape[0], y + bh + pad)
            zoom = highlight[y0:y1, x0:x1]
            
            st.image(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB), caption="選択穴を強調表示", use_container_width=True)
            st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB), caption=f"id={hole_id} の拡大", use_container_width=True)
            
            # CSVもid付きで保存される
            st.download_button(
                "CSVをダウンロード",
                df.to_csv(index=False).encode("utf-8"),
                file_name="afm_holes.csv",
                mime="text/csv"
            )
