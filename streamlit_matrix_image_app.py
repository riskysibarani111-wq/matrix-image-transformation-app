"""
Streamlit app: Transformasi Matriks & Konvolusi untuk Pemrosesan Gambar
File: streamlit_matrix_image_app.py
Dependencies: streamlit, numpy, opencv-python, matplotlib
Run: pip install streamlit numpy opencv-python matplotlib
     streamlit run streamlit_matrix_image_app.py

Deskripsi:
- Multi-halaman (sidebar navigation) dengan dua halaman: Geometri dan Konvolusi
- Implementasi transformasi matriks: translation, scaling, rotation, shearing, reflection
- Implementasi konvolusi dengan kernel kustom: blur (average) dan sharpening
- Upload gambar, set parameter, preview, dan download hasil

Catatan implementasi:
- Semua transformasi dibuat dari matriks affine 2x3 (atau 3x3 untuk komposisi)
- Konvolusi menggunakan cv2.filter2D dengan kernel buatan sendiri
"""

import io
import numpy as np
import cv2
import streamlit as st
from matplotlib import pyplot as plt

st.set_page_config(page_title="Transformasi Matriks & Konvolusi - Image App", layout="wide")

# --- Helper functions -------------------------------------------------

def load_image(uploaded_file):
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to_fit(img, max_size=800):
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# Build affine matrix from 3x3 homogeneous matrix for cv2.warpAffine
def affine_from_homography(H):
    # cv2.warpAffine expects 2x3 matrix
    return H[0:2, :]


def apply_affine(img, M, output_shape=None, border_value=(0, 0, 0)):
    h, w = img.shape[:2]
    if output_shape is None:
        output_shape = (w, h)
    dst = cv2.warpAffine(img, M, output_shape, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return dst


# Transformation matrices (3x3 homogeneous)
def translation_matrix(tx, ty):
    M = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    return M


def scaling_matrix(sx, sy, center=(0, 0)):
    cx, cy = center
    T1 = translation_matrix(-cx, -cy)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
    T2 = translation_matrix(cx, cy)
    return T2 @ S @ T1


def rotation_matrix(angle_deg, center=(0, 0)):
    angle = np.deg2rad(angle_deg)
    cos = np.cos(angle)
    sin = np.sin(angle)
    cx, cy = center
    T1 = translation_matrix(-cx, -cy)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    T2 = translation_matrix(cx, cy)
    return T2 @ R @ T1


def shearing_matrix(shx=0.0, shy=0.0, center=(0, 0)):
    cx, cy = center
    T1 = translation_matrix(-cx, -cy)
    Sh = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=np.float32)
    T2 = translation_matrix(cx, cy)
    return T2 @ Sh @ T1


def reflection_matrix(axis='x', center=(0, 0)):
    # axis: 'x', 'y', 'xy' (main diagonal), 'y=x' treated as 'xy', or 'origin'
    cx, cy = center
    T1 = translation_matrix(-cx, -cy)
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    elif axis == 'y':
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    elif axis in ('xy', 'diag', 'y=x'):
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    elif axis == 'origin':
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    else:
        R = np.eye(3, dtype=np.float32)
    T2 = translation_matrix(cx, cy)
    return T2 @ R @ T1


# Convolution kernels
def kernel_blur(size=3):
    assert size % 2 == 1 and size >= 1
    k = np.ones((size, size), dtype=np.float32) / (size * size)
    return k


def kernel_sharpen():
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return k


def apply_convolution(img, kernel):
    # Apply to each channel
    if img.ndim == 2:
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_DEFAULT)
    channels = cv2.split(img)
    out_channels = [cv2.filter2D(ch, -1, kernel, borderType=cv2.BORDER_DEFAULT) for ch in channels]
    return cv2.merge(out_channels)


# Utility: compose multiple transforms (3x3 homogeneous)
def compose_transforms(transforms):
    M = np.eye(3, dtype=np.float32)
    for T in transforms:
        M = T @ M
    return M


# Convert image to downloadable bytes
def image_to_bytes(img, fmt='PNG'):
    is_success, buffer = cv2.imencode('.' + fmt.lower(), img)
    io_buf = io.BytesIO(buffer)
    return io_buf


# --- Streamlit UI -----------------------------------------------------

st.title("Transformasi Matriks & Konvolusi — Aplikasi Demonstrasi")
st.markdown(
    "Aplikasi ini menunjukkan operasi transformasi matriks 2D (translation, scaling, rotation, shearing, reflection) dan operasi konvolusi (blur & sharpen) menggunakan kernel kustom. Built with Streamlit, NumPy, dan OpenCV."
)

page = st.sidebar.selectbox("Pilih halaman", ["Geometri (Transformasi)", "Konvolusi (Filtering)"])

uploaded = st.sidebar.file_uploader("Unggah gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    src_img = load_image(uploaded)
    if src_img is None:
        st.error("Gagal membaca file gambar. Pastikan file valid.")
        st.stop()
    src_img_disp = resize_to_fit(to_rgb(src_img), max_size=800)
else:
    st.info("Silakan unggah gambar di sidebar untuk memulai.")
    src_img = None
    src_img_disp = None


if page == "Geometri (Transformasi)":
    st.header("Transformasi Geometri (berbasis matriks)")
    st.write("Pilih operasi transformasi dan atur parameternya. Semua transformasi diterapkan sebagai matriks homogen 3x3.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transformasi Individu")
        op = st.selectbox("Pilih operasi", ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "Kombinasi"])

        # center default
        if src_img is not None:
            h, w = src_img.shape[:2]
            cx, cy = st.number_input("Center X (px)", value=int(w // 2)) , st.number_input("Center Y (px)", value=int(h // 2))
            center = (float(cx), float(cy))
        else:
            center = (0.0, 0.0)

        transforms = []
        if op == "Translation":
            tx = st.slider("Translation X (px)", -500, 500, 0)
            ty = st.slider("Translation Y (px)", -500, 500, 0)
            transforms = [translation_matrix(tx, ty)]
        elif op == "Scaling":
            sx = st.slider("Scale X", 0.1, 5.0, 1.0)
            sy = st.slider("Scale Y", 0.1, 5.0, 1.0)
            transforms = [scaling_matrix(sx, sy, center=center)]
        elif op == "Rotation":
            ang = st.slider("Angle (deg)", -180, 180, 0)
            transforms = [rotation_matrix(ang, center=center)]
        elif op == "Shearing":
            shx = st.slider("Shear X", -2.0, 2.0, 0.0)
            shy = st.slider("Shear Y", -2.0, 2.0, 0.0)
            transforms = [shearing_matrix(shx, shy, center=center)]
        elif op == "Reflection":
            axis = st.selectbox("Axis", ["x", "y", "xy", "origin"])
            transforms = [reflection_matrix(axis=axis, center=center)]
        elif op == "Kombinasi":
            st.markdown("Tambahkan beberapa transformasi berurutan (komposisi)")
            do_translate = st.checkbox("Tambah Translation")
            if do_translate:
                tx = st.number_input("tx (px)", value=0)
                ty = st.number_input("ty (px)", value=0)
            do_scale = st.checkbox("Tambah Scaling")
            if do_scale:
                sx = st.number_input("sx", value=1.0, format="%.3f")
                sy = st.number_input("sy", value=1.0, format="%.3f")
            do_rot = st.checkbox("Tambah Rotation")
            if do_rot:
                ang = st.number_input("angle (deg)", value=0)
            do_shear = st.checkbox("Tambah Shearing")
            if do_shear:
                shx = st.number_input("shx", value=0.0)
                shy = st.number_input("shy", value=0.0)
            do_reflect = st.checkbox("Tambah Reflection")
            if do_reflect:
                axis = st.selectbox("axis reflect", ["x", "y", "xy", "origin"], key="comb_ref")

            # build list in order
            transforms = []
            if do_translate:
                transforms.append(translation_matrix(tx, ty))
            if do_scale:
                transforms.append(scaling_matrix(sx, sy, center=center))
            if do_rot:
                transforms.append(rotation_matrix(ang, center=center))
            if do_shear:
                transforms.append(shearing_matrix(shx, shy, center=center))
            if do_reflect:
                transforms.append(reflection_matrix(axis=axis, center=center))

    with col2:
        st.subheader("Preview & Output")
        if src_img is None:
            st.warning("Unggah gambar untuk melihat preview")
        else:
            st.write("Gambar asli")
            st.image(src_img_disp, use_column_width=True)

            if len(transforms) == 0:
                st.info("Pilih transformasi di kolom kiri")
            else:
                M_h = compose_transforms(transforms)
                M_affine = affine_from_homography(M_h)

                # Determine output size: make same as input
                h, w = src_img.shape[:2]
                # Apply affine — warpAffine wants (width, height)
                dst = apply_affine(src_img, M_affine, output_shape=(w, h), border_value=(255, 255, 255))

                dst_disp = resize_to_fit(to_rgb(dst), max_size=800)
                st.write("Gambar setelah transformasi")
                st.image(dst_disp, use_column_width=True)

                buf = image_to_bytes(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), fmt='PNG')
                st.download_button("Download gambar hasil (PNG)", data=buf, file_name="transformed.png", mime="image/png")

                st.markdown("**Matriks homogen 3x3 hasil komposisi:**")
                st.code(np.array2string(M_h, precision=3, separator=', '))


elif page == "Konvolusi (Filtering)":
    st.header("Pemfilteran Gambar (Konvolusi dengan Kernel Kustom)")
    st.write("Contoh kernel: blur (average) dan sharpen — diimplementasikan menggunakan kernel buatan dan cv2.filter2D")

    if src_img is None:
        st.warning("Unggah gambar di sidebar untuk menggunakan fitur konvolusi")
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Pilih filter")
        filt = st.selectbox("Filter", ["Blur (Average)", "Gaussian-like Blur (custom)", "Sharpen (kernel)", "Custom Kernel"])

        if filt == "Blur (Average)":
            ksize = st.slider("Kernel size (odd)", 1, 31, 3, step=2)
            kernel = kernel_blur(ksize)
            st.write(f"Average kernel {ksize}x{ksize}")
            st.write(kernel)
        elif filt == "Gaussian-like Blur (custom)":
            ksize = st.slider("Kernel size (odd)", 3, 31, 5, step=2, key="gk")
            sigma = st.slider("Sigma", 0.1, 10.0, 1.0)
            # create separable gaussian kernel
            ax = cv2.getGaussianKernel(ksize, sigma)
            kernel = ax @ ax.T
            st.write(f"Gaussian-like kernel {ksize}x{ksize}")
            st.write(np.round(kernel, 4))
        elif filt == "Sharpen (kernel)":
            kernel = kernel_sharpen()
            st.write("Sharpening kernel")
            st.write(kernel)
        else:
            st.write("Masukkan kernel kustom (pisahkan elemen dengan koma, baris baru untuk baris baru)")
            txt = st.text_area("Kernel (contoh: 0,-1,0\n-1,5,-1\n0,-1,0)")
            try:
                rows = [list(map(float, row.split(','))) for row in txt.strip().splitlines() if row.strip()]
                kernel = np.array(rows, dtype=np.float32)
                st.write("Kernel yang dimasukkan:")
                st.write(kernel)
            except Exception:
                st.warning("Kernel tidak valid — gunakan format CSV per baris")
                kernel = None

        normalize = st.checkbox("Normalisasi kernel (jumlah elemen -> 1)", value=True)
        if kernel is not None and normalize:
            s = kernel.sum()
            if abs(s) > 1e-6:
                kernel = kernel / s

        apply_btn = st.button("Terapkan filter")

    with col2:
        st.subheader("Preview & Output")
        st.write("Gambar asli")
        st.image(resize_to_fit(to_rgb(src_img), 600), use_column_width=True)

        if not apply_btn:
            st.info("Tentukan filter lalu klik 'Terapkan filter'")
        else:
            if kernel is None:
                st.error("Kernel tidak valid—periksa input")
            else:
                dst = apply_convolution(src_img, kernel)
                dst = np.clip(dst, 0, 255).astype(np.uint8)
                st.write("Hasil setelah konvolusi")
                st.image(resize_to_fit(to_rgb(dst), 800), use_column_width=True)

                buf = image_to_bytes(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), fmt='PNG')
                st.download_button("Download hasil (PNG)", data=buf, file_name="convolved.png", mime="image/png")

                st.markdown("**Kernel yang diterapkan:**")
                st.code(np.array2string(kernel, precision=4, separator=', '))


# Footer
st.markdown("---")
st.markdown("**Catatan:** Semua operasi dijalankan pada CPU, dan implementasi konvolusi menggunakan `cv2.filter2D` dengan kernel kustom. Transformasi geometris disusun menggunakan matriks homogen 3x3 dan diturunkan ke bentuk 2x3 untuk `cv2.warpAffine`. Untuk hasil terbaik, gunakan gambar dengan resolusi sedang (tidak terlalu besar).")


