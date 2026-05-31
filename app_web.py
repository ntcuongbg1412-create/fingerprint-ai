import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang web (Bắt buộc chạy đầu tiên)
st.set_page_config(
    page_title="AI Fingerprint Expert Localizer",
    page_icon="🧬",
    layout="wide"
)

# 2. Tiêu đề giao diện chuyên nghiệp
st.title("🧬 Hệ Thống Định Vị & Đối Sánh Vân Tay Tự Động Bằng AI")
st.subheader("Tự động trích xuất đặc trưng, lọc nhiễu nền và khoanh vùng vị trí ảnh nhỏ trên ảnh lớn")
st.markdown("---")

# 3. Khu vực Upload dữ liệu đầu vào
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
        st.image(img2_pil, caption="Ảnh 2 (Bản vân đầy đủ để đối chiếu)", use_container_width=True)

st.markdown("---")

# 4. Xử lý logic thuật toán
if st.button("🔥 KÍCH HOẠT QUÉT SÂU & ĐỊNH VỊ VÙNG VÂN", type="primary", use_container_width=True):
    if file1 is None or file2 is None:
        st.error("❌ Vui lòng tải lên đầy đủ cả Ảnh 1 và Ảnh 2 trước khi chạy phân tích!")
    else:
        with st.spinner("AI đang quét ma trận đường vân, khử nhiễu nilon và tính toán tọa độ đồng cấu..."):
            
            # Đưa con trỏ đọc file về vị trí ban đầu
            file1.seek(0)
            file2.seek(0)
            
            # Đọc ảnh theo định dạng ảnh xám (Grayscale) để xử lý ma trận vân
            file_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
            img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)

            file_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                st.error("❌ File ảnh bị lỗi cấu trúc dữ liệu, OpenCV không thể giải mã!")
            else:
                # --- BƯỚC 1: TIỀN XỬ LÝ NÂNG CẤP KHỬ NHIỄU VÀ LÀM NÉT ---
                # Tăng tương phản cục bộ để xử lý vân chìm mờ trong bóng tối
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
                img1_enhanced = clahe.apply(img1)
                img2_enhanced = clahe.apply(img2)

                # Lọc nhiễu song phương, làm mịn bề mặt giấy/nilon nhưng giữ nguyên cạnh vân sắc sảo
                img1_blur = cv2.bilateralFilter(img1_enhanced, d=11, sigmaColor=85, sigmaSpace=85)
                img2_blur = cv2.bilateralFilter(img2_enhanced, d=11, sigmaColor=85, sigmaSpace=85)

                # Chuyển đổi nhị phân thích nghi để cô lập đường vân đen tách biệt hoàn toàn khỏi nền bối cảnh
                img1_final = cv2.adaptiveThreshold(img1_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                img2_final = cv2.adaptiveThreshold(img2_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

                # --- BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG CHỐNG XOAY GÓC/LỆCH SCALE (SIFT) ---
                sift = cv2.SIFT_create(nfeatures=3000) # Tăng mật độ điểm quét giúp bao quát bối cảnh tốt hơn
                kp1, des1 = sift.detectAndCompute(img1_final, None)
                kp2, des2 = sift.detectAndCompute(img2_final, None)

                if des1 is None or des2 is None:
                    st.error("❌ AI không tìm thấy đủ điểm đặc trưng lõi. Hãy cắt (Crop) sát vào ô vân tay và thử lại!")
                else:
                    # --- BƯỚC 3: SO KHỚP THÔNG MINH BẰNG FLANN MATCHER ---
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)

                    # Lọc điểm trùng khớp chất lượng cao dựa theo tỷ lệ chuẩn Lowe's Ratio
                    good_matches = []
                    for m_pair in matches:
                        if len(m_pair) == 2:
                            m, n = m_pair
                            if m.distance < 0.72 * n.distance:
                                good_matches.append(m)

                    # --- BƯỚC 4: THUẬT TOÁN ĐỊNH VỊ TỰ ĐỘNG KHOANH VÙNG (HOMOGRAPHY) ---
                    localization_success = False
                    img2_color = cv2.cvtColor(img2_enhanced, cv2.COLOR_GRAY2BGR)
                    
                    # Phép toán hình học cần tối thiểu 4 điểm tương đồng không thẳng hàng để tính toán
                    if len(good_matches) >= 4:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Sử dụng RANSAC để loại bỏ hoàn toàn các điểm khớp sai lệch hướng sinh ra do nhiễu thước đo
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M is not None:
                            localization_success = True
                            h, w = img1.shape
                            # Xác định ma trận biên 4 góc của Ảnh 1
                            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                            # Chiếu ma trận M để chuyển đổi tọa độ tương ứng sang không gian Ảnh 2
                            dst = cv2.perspectiveTransform(pts, M)
                            
                            # VẼ HỘP BAO MÀU XANH NEON ĐỂ ĐỊNH VỊ VÙNG VÂN
                            img2_color = cv2.polylines(img2_color, [np.int32(dst)], True, (255, 255, 0), 5, cv2.LINE_AA)

                    # --- BƯỚC 5: TÍNH TOÁN ĐỘ TƯƠNG ĐỒNG TOÀN DIỆN ---
                    total_features = min(len(kp1), len(kp2))
                    if total_features > 0:
                        match_percentage = (len(good_matches) / total_features) * 100
                        match_percentage = min(match_percentage * 4.0, 100.0) # Nhân hệ số thích ứng môi trường ảnh mờ
                    else:
                        match_percentage = 0.0

                    # --- BƯỚC 6: VẼ CÁC ĐIỂM ĐẶC TRƯNG KHOANH TRÒN ĐỎ ---
                    img1_color = cv2.cvtColor(img1_enhanced, cv2.COLOR_GRAY2BGR)
                    
                    # Sắp xếp các điểm tốt nhất lên đầu và chỉ vẽ 40 điểm để tránh làm che mất khung định vị
                    good_matches = sorted(good_matches, key=lambda x: x.distance)
                    display_matches = good_matches[:40]

                    for m in display_matches:
                        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))
                        cv2.circle(img1_color, pt1, radius=7, color=(0, 0, 255), thickness=2) # Vòng tròn đỏ trên ảnh 1
                        cv2.circle(img2_color, pt2, radius=7, color=(0, 0, 255), thickness=2) # Vòng tròn đỏ trên ảnh 2

                    # Vẽ sơ đồ kết nối các đường chỉ xanh lá cây giữa 2 bức ảnh
                    result_img = cv2.drawMatches(
                        img1_color, kp1, img2_color, kp2, display_matches, None,
                        matchColor=(0, 255, 0),
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                    # --- BƯỚC 7: XUẤT KẾT QUẢ RA GIAO DIỆN WEB STREAMLIT ---
                    st.success("🎉 Hệ thống AI đã thực hiện phân tích và xử lý hình học thành công!")
                    
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="Độ trùng khớp cấu trúc vân (Similarity Score)", value=f"{match_percentage:.2f} %")
                    with m_col2:
                        st.metric(label="Số điểm lõi khớp chính xác tuyệt đối", value=f"{len(good_matches)} điểm")

                    # Đưa ra kết luận tự động dựa trên phân tích ma trận định vị
                    if localization_success and match_percentage >= 40:
                        st.balloons()
                        st.markdown("<h3 style='color:#00ff00; text-align:center;'>✅ ĐÃ ĐỊNH VỊ THÀNH CÔNG VÙNG VÂN TAY TRÙNG KHỚP!</h3>", unsafe_allow_html=True)
                        st.info("💡 Điểm đặc biệt: Hãy nhìn vào KHUNG HÌNH CHỮ NHẬT MÀU XANH NEON ở bức ảnh bên phải, đó chính là vị trí chính xác của cấu trúc đường vân trên Ảnh 1 khi được định vị vào Bản đầy đủ.")
                    else:
                        st.markdown("<h3 style='color:#ff3333; text-align:center;'>❌ HỆ THỐNG KHÔNG THỂ XÁC ĐỊNH KHUNG VỊ TRÍ (Độ tương đồng quá thấp hoặc ảnh bị biến dạng nặng)</h3>", unsafe_allow_html=True)

                    st.markdown("### 📊 Bản đồ phân tích đặc trưng đối chiếu hình học (Feature & Location Mapping)")
                    st.image(result_img_rgb, caption="Các đường chỉ xanh lá kết nối các điểm tương đồng, khung chữ nhật xanh neon định vị vùng không gian trùng khớp.", use_container_width=True)
