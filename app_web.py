import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang web (Chạy đầu tiên)
st.set_page_config(
    page_title="AI Fingerprint Analyzer",
    page_icon="🧬",
    layout="wide"
)

# 2. Tiêu đề giao diện
st.title("🧬 Hệ Thống Phân Tích & So Khớp Vân Tay Bằng AI")
st.subheader("Tải lên 2 ảnh vân tay để tìm các điểm tương đồng (Hỗ trợ ảnh mờ, nhòe)")
st.markdown("---")

# 3. Tạo layout 2 cột để người dùng upload ảnh
col1, col2 = st.columns(2)

with col1:
    st.header("Vân tay thứ nhất")
    file1 = st.file_uploader("Chọn ảnh vân tay 1...", type=["jpg", "jpeg", "png", "bmp"], key="file1")
    if file1:
        img1_pil = Image.open(file1)
        # Sửa use_column_width thành use_container_width để hết cảnh báo vàng
        st.image(img1_pil, caption="Ảnh 1 gốc", use_container_width=True)

with col2:
    st.header("Vân tay thứ hai")
    file2 = st.file_uploader("Chọn ảnh vân tay 2...", type=["jpg", "jpeg", "png", "bmp"], key="file2")
    if file2:
        img2_pil = Image.open(file2)
        st.image(img2_pil, caption="Ảnh 2 gốc", use_container_width=True)

st.markdown("---")

# 4. Xử lý thuật toán khi người dùng nhấn nút Phân tích
if st.button("🔥 BẮT ĐẦU PHÂN TÍCH SO KHỚP", type="primary", use_container_width=True):
    if file1 is None or file2 is None:
        st.error("❌ Vui lòng tải lên đầy đủ cả 2 ảnh vân tay trước khi phân tích!")
    else:
        # Hiển thị hiệu ứng loading xoay tròn chuyên nghiệp của Web
        with st.spinner("AI đang lọc nhiễu, làm nét đường vân và đối sánh..."):
            
            # ĐƯA CON TRỎ ĐỌC VỀ ĐẦU FILE ĐỂ TRÁNH LỖI BUF.EMPTY()
            file1.seek(0)
            file2.seek(0)
            
            # Chuyển ảnh từ định dạng định dạng Streamlit sang cấu trúc OpenCV (Numpy Array)
            file_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
            img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)

            file_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                st.error("❌ Không thể đọc được dữ liệu hình ảnh. Vui lòng kiểm tra lại định dạng file ảnh của bạn!")
            else:
                # --- TIỀN XỬ LÝ NÂNG CẤP (Xử lý ảnh mờ, nhòe) ---
                # 1. Tăng cường độ tương phản bằng CLAHE
                clahe = cv2.createCLEHE = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                img1_enhanced = clahe.apply(img1)
                img2_enhanced = clahe.apply(img2)

                # 2. Khử nhiễu hạt và làm mịn các cạnh vân bị nhòe
                img1_blur = cv2.bilateralFilter(img1_enhanced, d=9, sigmaColor=75, sigmaSpace=75)
                img2_blur = cv2.bilateralFilter(img2_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

                # 3. Kỹ thuật chuyển đổi làm rõ nét đường vân (Adaptive Thresholding)
                img1_final = cv2.adaptiveThreshold(img1_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                img2_final = cv2.adaptiveThreshold(img2_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # --- AI TRÍCH XUẤT ĐẶC TRƯNG (ORB) ---
                orb = cv2.ORB_create(nfeatures=1500)
                kp1, des1 = orb.detectAndCompute(img1_final, None)
                kp2, des2 = orb.detectAndCompute(img2_final, None)

                if des1 is None or des2 is None:
                    st.error("❌ AI không thể tìm thấy đủ điểm đặc trưng trên một trong hai bức ảnh. Hãy thử ảnh rõ nét hơn!")
                else:
                    # --- SO KHỚP ĐIỂM (Brute-Force Matcher) ---
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                    # Lọc các điểm khớp có chất lượng tốt
                    good_matches = [m for m in matches if m.distance < 45]

                    # --- TÍNH PHẦN TRĂM ĐỘ TƯƠNG ĐỒNG ---
                    total_features = min(len(kp1), len(kp2))
                    if total_features > 0:
                        match_percentage = (len(good_matches) / total_features) * 100
                        match_percentage = min(match_percentage * 2.5, 100.0)
                    else:
                        match_percentage = 0.0

                    # --- VẼ BẢN ĐỒ ĐIỂM GIỐNG NHAU ---
                    result_img = cv2.drawMatches(
                        img1_enhanced, kp1, img2_enhanced, kp2, good_matches[:40], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                    # --- HIỂN THỊ KẾT QUẢ LÊN GIAO DIỆN WEB ---
                    st.success("🎉 Phân tích hoàn tất!")
                    
                    stat1, stat2 = st.columns(2)
                    with stat1:
                        st.metric(label="Độ tương đồng (Similarity Score)", value=f"{match_percentage:.2f} %")
                    with stat2:
                        st.metric(label="Số điểm giống nhau tìm thấy", value=f"{len(good_matches)} điểm")

                    if match_percentage >= 75:
                        st.balloons()
                        st.markdown("<h3 style='color:#00ff00; text-align:center;'>KẾT LUẬN: ĐỒNG NHẤT (Khả năng cao cùng 1 người)</h3>", unsafe_allow_html=True)
                    elif match_percentage >= 45:
                        st.markdown("<h3 style='color:#ffcc00; text-align:center;'>KẾT LUẬN: CÓ THỂ GIỐNG (Cần thêm ảnh rõ hơn để xác minh)</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='color:#ff3333; text-align:center;'>KẾT LUẬN: KHÔNG TRÙNG KHỚP</h3>", unsafe_allow_html=True)

                    st.markdown("### 📊 Bản đồ đối chiếu các điểm đặc trưng giống nhau (Feature Keypoints Mapping)")
                    st.image(result_img_rgb, caption="Các đường nối thể hiện các vị trí có cấu trúc vân tương đồng nhau", use_container_width=True)