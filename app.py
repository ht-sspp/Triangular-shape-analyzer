import streamlit as st
from PIL import Image
import numpy as np
import cv2

from streamlit_cropper import st_cropper
from afm_analysis import analyze_triangle_holes

st.set_page_config(page_title="AFMメタサーフェス自動解析", layout="wide")
st.title("AFMメタサーフェス（三角穴）自動解析")

st.write("1) 画像をアップロード → 2) パターン領域だけドラッグで切り出し → 3) 切り出し領域の実サイズ(µm)を入力 → 4) 解析")

uploaded = st.file_uploader("AFM画像（png/jpg/tif/bmp）", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.sidebar.header("解析設定")
    field_um = st.sidebar.number_input(
        "切り出し領域の一辺の長さ [µm]（正方形スキャン想定）",
        min_value=0.01, value=5.0, step=0.1
    )
    min_area_px = st.sidebar.number_input("最小穴面積しきい値 [px^2]（ノイズ除去）", min_value=0, value=50, step=10)
    blur_sigma = st.sidebar.number_input("平滑化 sigma", min_value=0.0, value=1.0, step=0.5)

    st.subheader("① 解析したい領域を切り出し（ファイル情報の表示部分を除外）")
    cropped = st_cropper(img, realtime_update=True, aspect_ratio=(1, 1))
    st.caption("※ 正方形で切り出すのがラクです（スキャン領域が正方形の場合）")

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
            summary, df, bin_img, overlay = result

            st.subheader("サマリー")
            st.write(f"- 検出穴数: **{summary['n']}**")
            st.write(f"- 一辺（面積から換算）: **{summary['L_mean_um']:.3f} ± {summary['L_std_um']:.3f} µm**")
            st.write(f"- 周期（最近接距離の代表値）: **{summary['pitch_um']:.3f} µm**")
            st.write(f"- スケール: **{summary['um_per_px']:.6f} µm/px**")

            c3, c4 = st.columns(2)
            with c3:
                st.image(bin_img, caption="2値化（穴抽出）", use_container_width=True)
            with c4:
                st.image(overlay, caption="検出結果（輪郭+重心）", use_container_width=True)

            st.subheader("穴ごとの結果（CSVダウンロード可）")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "CSVをダウンロード",
                df.to_csv(index=False).encode("utf-8"),
                file_name="afm_holes.csv",
                mime="text/csv"
            )
