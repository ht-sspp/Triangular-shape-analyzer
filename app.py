from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw
import numpy as np
import cv2
import streamlit as st

st.set_page_config(page_title="AFMメタサーフェス自動解析", layout="wide")
st.title("AFMメタサーフェス（三角穴）自動解析")

st.write("1) 画像をアップロード → 2) パターン領域だけドラッグで切り出し → 3) 切り出し領域の実サイズ(µm)を入力 → 4) 解析")

uploaded = st.file_uploader(
    "AFM画像（png/jpg/bmp/tif）",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.subheader("① 解析したい領域をドラッグで選択（正方形推奨）")
    box = st_cropper(
        img,
        realtime_update=True,
        aspect_ratio=(1, 1),
        return_type="box",  # ←ここが重要（座標を返す）  [oai_citation:1‡Streamlit](https://discuss.streamlit.io/t/how-to-get-back-the-x-y-coordinates-with-st-cropper/46968?utm_source=chatgpt.com)
    )

    # box は {"left":..., "top":..., "width":..., "height":...}
    left = int(box["left"]); top = int(box["top"])
    width = int(box["width"]); height = int(box["height"])
    right = left + width; bottom = top + height

    # 元画像にROI枠を描画（見やすさアップ）
    img_with_box = img.copy()
    draw = ImageDraw.Draw(img_with_box)
    draw.rectangle([left, top, right, bottom], outline=(0, 255, 255), width=4)  # シアン枠

    cropped = img.crop((left, top, right, bottom))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_with_box, caption="元画像（ROI枠付き）", use_container_width=True)
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
