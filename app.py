import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2

from streamlit_cropper import st_cropper
from afm_analysis import analyze_triangle_holes

st.set_page_config(page_title="AFMメタサーフェス自動解析", layout="wide")
st.title("AFMメタサーフェス（三角穴）自動解析")
st.write("1) 画像をアップロード → 2) パターン領域だけドラッグで切り出し → 3) スケール指定 → 4) 解析")

uploaded = st.file_uploader(
    "AFM画像（png/jpg/tif/bmp）",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # ------------------------
    # Sidebar: settings
    # ------------------------
    st.sidebar.header("解析設定")

    scale_mode = st.sidebar.radio(
        "スケール指定方法",
        ["切り出し領域の一辺 [µm] を入力", "1ピクセルあたりの長さ [µm/px] を入力"]
    )

    field_um_input = None
    um_per_px_input = None

    if scale_mode == "切り出し領域の一辺 [µm] を入力":
        field_um_input = st.sidebar.number_input(
            "切り出し領域の一辺 [µm]",
            min_value=0.01, value=5.0, step=0.1
        )
    else:
        um_per_px_input = st.sidebar.number_input(
            "1 px あたりの長さ [µm/px]",
            min_value=1e-6, value=0.01, step=0.001, format="%.6f"
        )

    min_area_px = st.sidebar.number_input(
        "最小穴面積しきい値 [px^2]（ノイズ除去）",
        min_value=0, value=50, step=10
    )
    blur_sigma = st.sidebar.number_input(
        "平滑化 sigma",
        min_value=0.0, value=1.0, step=0.5
    )

    # ------------------------
    # Crop UI
    # ------------------------
    st.subheader("① 解析したい領域を切り出し（ファイル情報の表示部分を除外）")

    # ROI座標を得る（どこを切ったか見やすくする）
    box = st_cropper(
        img,
        realtime_update=True,
        aspect_ratio=(1, 1),
        return_type="box",
    )

    left = int(box["left"])
    top = int(box["top"])
    width = int(box["width"])
    height = int(box["height"])
    right = left + width
    bottom = top + height

    # 安全対策（ゼロサイズ回避）
    if width <= 1 or height <= 1:
        st.warning("切り出し領域が小さすぎます。もう少し大きく選択してください。")
        st.stop()

    # ROI枠付きの元画像（見やすさ）
    img_with_box = img.copy()
    draw = ImageDraw.Draw(img_with_box)
    draw.rectangle([left, top, right, bottom], outline=(0, 255, 255), width=4)

    # Crop実行
    cropped = img.crop((left, top, right, bottom))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_with_box, caption="元画像（ROI枠付き）", use_container_width=True)
    with col2:
        st.image(cropped, caption="切り出し後（解析対象）", use_container_width=True)

    # ------------------------
    # Compute scale: um_per_px
    # ------------------------
    crop_w_px, crop_h_px = cropped.size  # PILは (W,H)
    if scale_mode == "切り出し領域の一辺 [µm] を入力":
        um_per_px = float(field_um_input) / float(crop_w_px)
    else:
        um_per_px = float(um_per_px_input)

    st.caption(f"スケール: {um_per_px:.6f} µm/px（切り出し幅 {crop_w_px}px）")

    # ------------------------
    # Run analysis
    # ------------------------
    if st.button("② 解析する"):
        # PIL -> OpenCV (BGR)
        cropped_np = np.array(cropped)  # RGB
        cropped_bgr = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)

        result = analyze_triangle_holes(
            cropped_bgr,
            um_per_px=um_per_px,
            min_area_px=int(min_area_px),
            blur_sigma=float(blur_sigma),
        )

        if result is None:
            st.error("穴が検出できませんでした。min_area / sigma を調整するか、切り出し範囲を見直してください。")
            st.stop()

        summary, df, bin_img, overlay_rgb, kept_contours, centers_px = result

        st.subheader("サマリー")
        st.write(f"- 検出穴数: **{summary['n']}**")
        st.write(f"- 一辺（面積換算）: **{summary['L_mean_um']:.3f} ± {summary['L_std_um']:.3f} µm**")
        st.write(f"- 周期（最近接距離の代表値）: **{summary['pitch_um']:.3f} µm**")
        st.write(f"- um_per_px: **{summary['um_per_px']:.6f} µm/px**")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("2値化（穴抽出）")
            st.image(bin_img, use_container_width=True)
        with c4:
            st.subheader("検出結果（番号＝表の id）")
            st.image(overlay_rgb, use_container_width=True)

        st.subheader("穴ごとの結果（idで画像と対応）")
        # 表示したい列だけ（必要に応じて追加）
        show_cols = ["id", "L_um", "area_um2", "cx_um", "cy_um"]
        if all(c in df.columns for c in show_cols):
            st.dataframe(df[show_cols], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

        # # ---- 選んだ穴を強調表示＆拡大表示 ----
        # st.subheader("特定の穴を選んで確認")
        # hole_id = st.selectbox("hole id", df["id"].tolist(), index=0)

        # # 強調オーバーレイ（切り出し画像上）
        # highlight = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR).copy()
        # cv2.drawContours(highlight, kept_contours, -1, (80, 80, 80), 1)  # 全輪郭薄く

        # c_sel = kept_contours[int(hole_id)]
        # cv2.drawContours(highlight, [c_sel], -1, (0, 0, 255), 3)  # 選択穴太く

        # # bboxで拡大（dfにbbox列がある前提）
        # if all(k in df.columns for k in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]):
        #     row = df[df["id"] == hole_id].iloc[0]
        #     x, y, bw, bh = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        #     pad = 20
        #     x0 = max(0, x - pad); y0 = max(0, y - pad)
        #     x1 = min(highlight.shape[1], x + bw + pad); y1 = min(highlight.shape[0], y + bh + pad)
        #     zoom = highlight[y0:y1, x0:x1]
        # else:
        #     zoom = highlight  # bboxが無ければ全体

        # st.image(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB), caption="選択穴を強調表示", use_container_width=True)
        # st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB), caption=f"id={hole_id} の拡大", use_container_width=True)

        st.download_button(
            "CSVをダウンロード",
            df.to_csv(index=False).encode("utf-8"),
            file_name="afm_holes.csv",
            mime="text/csv"
        )
