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
        # Hiển thị hiệu ứng loading xoay tròn
        with st.spinner("AI đang lọc nhiễu, làm nét đường vân và đối sánh..."):
            
            # Đưa con trỏ đọc về đầu file để tránh lỗi empty buf
            file1.seek(0)
            file2.seek(0)
            
            # Chuyển ảnh sang cấu trúc OpenCV (Màu xám)
            file_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
            img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)

            file_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                st.error("❌ Không thể đọc được dữ liệu hình ảnh. Vui lòng kiểm tra lại định dạng file ảnh của bạn!")
            else:
                # --- TIỀN XỬ LÝ NÂNG CẤP (Xử lý ảnh mờ, nhòe) ---
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                img1_enhanced = clahe.apply(img1)
                img2_enhanced = clahe.apply(img2)

                img1_blur = cv2.bilateralFilter(img1_enhanced, d=9, sigmaColor=75, sigmaSpace=75)
                img2_blur = cv2.bilateralFilter(img2_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

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

                    # Giới hạn hiển thị tối đa 40 điểm tốt nhất để không bị rối mắt
                    display_matches = good_matches[:40]

                    # --- TÍNH PHẦN TRĂM ĐỘ TƯƠNG ĐỒNG ---
                    total_features = min(len(kp1), len(kp2))
                    if total_features > 0:
                        match_percentage = (len(good_matches) / total_features) * 100
                        match_percentage = min(match_percentage * 2.5, 100.0)
                    else:
                        match_percentage = 0.0

                    # --- CHUYỂN ẢNH GỐC SANG MÀU ĐỂ VẼ VÒNG TRÒN ĐỎ ---
                    img1_color = cv2.cvtColor(img1_enhanced, cv2.COLOR_GRAY2BGR)
                    img2_color = cv2.cvtColor(img2_enhanced, cv2.COLOR_GRAY2BGR)

                    # Tiến hành khoanh tròn đỏ vào các điểm khớp trên Ảnh 1 và Ảnh 2
                    for m in display_matches:
                        # Lấy tọa độ điểm trên ảnh 1
                        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                        # Lấy tọa độ điểm trên ảnh 2
                        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))
                        
                        # Vẽ vòng tròn màu đỏ (BGR: 0, 0, 255), độ dày viền là 2, bán kính là 8
                        cv2.circle(img1_color, pt1, radius=8, color=(0, 0, 255), thickness=2)
                        cv2.circle(img2_color, pt2, radius=8, color=(0, 0, 255), thickness=2)

                    # --- VẼ BẢN ĐỒ ĐIỂM GIỐNG NHAU (Nối các điểm đã khoanh tròn) ---
                    result_img = cv2.drawMatches(
                        img1_color, kp1, img2_color, kp2, display_matches, None,
                        matchColor=(0, 255, 0), # Đường nối màu xanh lá cho nổi bật
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
                    st.image(result_img_rgb, caption="Các vòng tròn đỏ thể hiện vị trí điểm tương đồng, đường xanh nối các điểm trùng khớp giữa 2 ảnh", use_container_width=True)
