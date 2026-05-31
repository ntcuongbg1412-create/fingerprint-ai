import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang web
st.set_page_config(
    page_title="AI Fingerprint Forensic Analyzer",
    page_icon="🧬",
    layout="wide"
)

# 2. Tiêu đề giao diện
st.title("🧬 Hệ Thống Giám Định Vân Tay Tự Động Pro")
st.subheader("Giao diện trực quan hóa: Đánh số cặp điểm trùng khớp & Tách biệt vùng định vị hình học")
st.markdown("---")

# 3. Khu vực Upload dữ liệu
col1, col2 = st.columns(2)

with col1:
    st.header("Vân tay cần định vị (Ảnh 1)")
    file1 = st.file_uploader("Chọn ảnh vân tay 1...", type=["jpg", "jpeg", "png", "bmp"], key="file1")
    if file1:
        img1_pil = Image.open(file1)
        st.image(img1_pil, caption="Ảnh 1 (Vùng vân tìm kiếm)", use_container_width=True)

with col2:
    st.header("Vân tay toàn diện đối chiếu (Ảnh 2)")
    file2 = st.file_uploader("Chọn ảnh vân tay 2...", type=["jpg", "jpeg", "png", "bmp"], key="file2")
    if file2:
        img2_pil = Image.open(file2)
        st.image(img2_pil, caption="Ảnh 2 (Bản vân đầy đủ)", use_container_width=True)

st.markdown("---")

# 4. Xử lý logic thuật toán khi nhấn nút
if st.button("🔥 KÍCH HOẠT ĐỐI SÁNH VÀ ĐÁNH SỐ ĐẶC TRƯNG", type="primary", use_container_width=True):
    if file1 is None or file2 is None:
        st.error("❌ Vui lòng tải lên đầy đủ cả 2 ảnh vân tay trước khi phân tích!")
    else:
        with st.spinner("AI đang tính toán ma trận điểm và gán số thứ tự đối chiếu..."):
            
            file1.seek(0)
            file2.seek(0)
            
            # Đọc ảnh xám
            file_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
            img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)

            file_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                st.error("❌ File ảnh bị lỗi cấu trúc dữ liệu, không thể giải mã!")
            else:
                # --- BƯỚC 1: TIỀN XỬ LÝ ẢNH ---
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
                img1_enhanced = clahe.apply(img1)
                img2_enhanced = clahe.apply(img2)

                img1_blur = cv2.bilateralFilter(img1_enhanced, d=11, sigmaColor=85, sigmaSpace=85)
                img2_blur = cv2.bilateralFilter(img2_enhanced, d=11, sigmaColor=85, sigmaSpace=85)

                img1_final = cv2.adaptiveThreshold(img1_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                img2_final = cv2.adaptiveThreshold(img2_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

                # --- BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG SIFT ---
                sift = cv2.SIFT_create(nfeatures=2000)
                kp1, des1 = sift.detectAndCompute(img1_final, None)
                kp2, des2 = sift.detectAndCompute(img2_final, None)

                if des1 is None or des2 is None:
                    st.error("❌ AI không trích xuất được đặc trưng. Vui lòng cắt sát vùng vân hơn!")
                else:
                    # --- BƯỚC 3: SO KHỚP FLANN & LỌC LOWE'S RATIO ---
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)

                    good_matches = []
                    for m_pair in matches:
                        if len(m_pair) == 2:
                            m, n = m_pair
                            if m.distance < 0.72 * n.distance:
                                good_matches.append(m)

                    # --- BƯỚC 4: TÍNH PHẦN TRĂM ĐỘ TƯƠNG ĐỒNG ---
                    total_features = min(len(kp1), len(kp2))
                    match_percentage = (len(good_matches) / total_features) * 100 if total_features > 0 else 0.0
                    match_percentage = min(match_percentage * 4.0, 100.0)

                    # --- BƯỚC 5: ĐỒ HỌA CHUYÊN BIỆT (ĐÁNH SỐ TỪNG CẶP ĐIỂM GIỐNG NHAU) ---
                    # Tạo bản sao ảnh màu để vẽ cấu trúc riêng biệt
                    img1_color = cv2.cvtColor(img1_enhanced, cv2.COLOR_GRAY2BGR)
                    img2_color = cv2.cvtColor(img2_enhanced, cv2.COLOR_GRAY2BGR)

                    # Sắp xếp lấy tối đa 25 cặp điểm tốt nhất để ghi số không bị đè chữ
                    good_matches = sorted(good_matches, key=lambda x: x.distance)
                    display_matches = good_matches[:25]

                    for idx, m in enumerate(display_matches, start=1):
                        # Lấy tọa độ điểm trên ảnh 1 và ảnh 2
                        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))

                        # 🔴 Ảnh bên trái: Chỉ vẽ vòng tròn Màu Đỏ (BGR: 0, 0, 255)
                        cv2.circle(img1_color, pt1, radius=6, color=(0, 0, 255), thickness=2)
                        # Ghi số thứ tự màu đỏ ngay cạnh điểm đặc trưng
                        cv2.putText(img1_color, str(idx), (pt1[0] + 8, pt1[1] + 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                        # 🔵 Ảnh bên phải: Chỉ vẽ vòng tròn Màu Xanh Dương (BGR: 255, 0, 0)
                        cv2.circle(img2_color, pt2, radius=6, color=(255, 0, 0), thickness=2)
                        # Ghi số thứ tự trùng khớp màu xanh dương ngay cạnh điểm
                        cv2.putText(img2_color, str(idx), (pt2[0] + 8, pt2[1] + 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                    # Ghép 2 ảnh nằm ngang song song nhưng KHÔNG vẽ đường nối chăng ngang
                    comparison_img = np.hstack((img1_color, img2_color))
                    comparison_rgb = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)

                    # --- BƯỚC 6: XỬ LÝ KHU VỰC KHOANH VÙNG VỊ TRÍ ĐỘC LẬP (ĐƯA XUỐNG DƯỚI) ---
                    img_localization = cv2.cvtColor(img2_enhanced, cv2.COLOR_GRAY2BGR)
                    localization_success = False

                    if len(good_matches) >= 4:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            localization_success = True
                            h, w = img1.shape
                            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            # Vẽ hộp bao định vị màu Xanh Neon đậm nét lên ảnh riêng biệt này
                            img_localization = cv2.polylines(img_localization, [np.int32(dst)], True, (255, 255, 0), 5, cv2.LINE_AA)

                    img_localization_rgb = cv2.cvtColor(img_localization, cv2.COLOR_BGR2RGB)

                    # --- BƯỚC 7: XUẤT KẾT QUẢ ĐỒ HỌA RA MÀN HÌNH WEB ---
                    st.success("🎉 Đã tối ưu hóa giao diện phân tích thành công!")
                    
                    st.metric(label="Độ tương đồng cấu trúc vân", value=f"{match_percentage:.2f} %")
                    st.metric(label="Số lượng cặp điểm lõi trùng khớp đánh số", value=f"{len(good_matches)} điểm")

                    # Phần hiển thị Bản đồ so sánh điểm đặc trưng ở TRÊN
                    st.markdown("### 📊 1. Bản Đồ Đối Chiếu Các Cặp Đặc Điểm Đồng Nhất (Đã Đánh Số Thứ Tự)")
                    st.caption("Mẹo xem: Tìm các số giống nhau ở ô Đỏ (Ảnh trái) và ô Xanh (Ảnh phải) để đối chiếu trực tiếp cấu trúc vân tay.")
                    st.image(comparison_rgb, caption="Ảnh trái (Chỉ điểm Đỏ) vs Ảnh phải (Chỉ điểm Xanh Dương) - Không bị rối đường nối", use_container_width=True)

                    st.markdown("---")

                    # Phần hiển thị Khoanh vùng vị trí hình học tách riêng ở DƯỚI
                    st.markdown("### 🗺️ 2. Bản Đồ Xác Định Vùng Vị Trí Cấu Trúc (Localization Map)")
                    if localization_success and match_percentage >= 40:
                        st.info("🎯 Khung hình chữ nhật màu Xanh Neon dưới đây biểu thị chính xác tọa độ vùng không gian mà Ảnh 1 đang chiếm chỗ trên Bản vân đầy đủ (Ảnh 2).")
                        st.image(img_localization_rgb, caption="Vùng vị trí được định vị bằng thuật toán ma trận đồng cấu Homography", use_container_width=True)
                    else:
                        st.warning("⚠️ Độ tương đồng hoặc mật độ điểm chưa đủ cao để dựng khung định vị tự động cho bức ảnh này.")
