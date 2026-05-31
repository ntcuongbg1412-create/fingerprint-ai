import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang web đầu tiên
st.set_page_config(
    page_title="AI Fingerprint Forensic Mapping",
    page_icon="🧬",
    layout="wide"
)

# 2. Tiêu đề giao diện
st.title("🧬 Hệ Thống Giám Định & Định Vị Vân Tay Hình Sự")
st.subheader("Giao diện tối ưu: Đánh số cặp điểm đối chiếu độc lập và Khử hoàn toàn đường nối rối mắt")
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
        st.image(img2_pil, caption="Ảnh 2 (Bản vân đầy đủ)", use_container_width=True)

st.markdown("---")

# 4. Xử lý logic thuật toán
if st.button("🔥 KÍCH HOẠT ĐỐI SÁNH VÀ ĐÁNH SỐ ĐẶC TRƯNG", type="primary", use_container_width=True):
    if file1 is None or file2 is None:
        st.error("❌ Vui lòng tải lên đầy đủ cả 2 ảnh vân tay trước khi phân tích!")
    else:
        with st.spinner("AI đang quét ma trận đường vân và đồng bộ hóa số thứ tự cặp điểm..."):
            
            file1.seek(0)
            file2.seek(0)
            
            # Đọc ảnh grayscale
            file_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
            img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)

            file_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                st.error("❌ Không thể giải mã tệp ảnh. Vui lòng kiểm tra lại định dạng file!")
            else:
                # --- BƯỚC 1: TIỀN XỬ LÝ ẢNH NÂNG CẤP ---
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
                img1_enhanced = clahe.apply(img1)
                img2_enhanced = clahe.apply(img2)

                img1_blur = cv2.bilateralFilter(img1_enhanced, d=11, sigmaColor=85, sigmaSpace=85)
                img2_blur = cv2.bilateralFilter(img2_enhanced, d=11, sigmaColor=85, sigmaSpace=85)

                img1_final = cv2.adaptiveThreshold(img1_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                img2_final = cv2.adaptiveThreshold(img2_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

                # --- BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG SIFT ---
                sift = cv2.SIFT_create(nfeatures=2500)
                kp1, des1 = sift.detectAndCompute(img1_final, None)
                kp2, des2 = sift.detectAndCompute(img2_final, None)

                if des1 is None or des2 is None:
                    st.error("❌ Không đủ điểm đặc trưng. Hãy thử một ảnh cắt sát đường vân rõ nét hơn!")
                else:
                    # --- BƯỚC 3: SO KHỚP FLANN & LỌC ĐIỂM CHUẨN ---
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

                    # Tính phần trăm tương đồng
                    total_features = min(len(kp1), len(kp2))
                    match_percentage = (len(good_matches) / total_features) * 100 if total_features > 0 else 0.0
                    match_percentage = min(match_percentage * 4.0, 100.0)

                    # --- BƯỚC 4: VẼ ĐIỂM ĐẶC TRƯNG VÀ GÁN SỐ THỨ TỰ ĐỒNG NHẤT ---
                    # Chuyển ảnh tăng cường sang ảnh màu BGR để vẽ ký hiệu màu sắc
                    out_img1 = cv2.cvtColor(img1_enhanced, cv2.COLOR_GRAY2BGR)
                    out_img2 = cv2.cvtColor(img2_enhanced, cv2.COLOR_GRAY2BGR)

                    # Sắp xếp và lấy ra 20 cặp điểm trùng khớp có độ chính xác cao nhất
                    good_matches = sorted(good_matches, key=lambda x: x.distance)
                    display_matches = good_matches[:20]

                    for idx, m in enumerate(display_matches, start=1):
                        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))

                        # 🔴 Ảnh bên trái (Ảnh 1): Chỉ vẽ vòng tròn và số Màu Đỏ
                        cv2.circle(out_img1, pt1, radius=6, color=(0, 0, 255), thickness=2)
                        cv2.putText(out_img1, str(idx), (pt1[0] + 8, pt1[1] + 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                        # 🔵 Ảnh bên phải (Ảnh 2): Chỉ vẽ vòng tròn và số Màu Xanh Dương (BGR: 255, 0, 0)
                        cv2.circle(out_img2, pt2, radius=6, color=(255, 0, 0), thickness=2)
                        cv2.putText(out_img2, str(idx), (pt2[0] + 8, pt2[1] + 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

                    # Chuyển đổi định dạng màu sang RGB để Streamlit hiển thị chính xác
                    out_img1_rgb = cv2.cvtColor(out_img1, cv2.COLOR_BGR2RGB)
                    out_img2_rgb = cv2.cvtColor(out_img2, cv2.COLOR_BGR2RGB)

                    # --- BƯỚC 5: TÍNH TOÁN KHU VỰC KHOANH VÙNG ĐỊNH VỊ TỰ ĐỘNG ---
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
                            # Vẽ hộp chữ nhật bao màu Xanh Neon lên ảnh bối cảnh đầy đủ
                            img_localization = cv2.polylines(img_localization, [np.int32(dst)], True, (255, 255, 0), 5, cv2.LINE_AA)

                    img_localization_rgb = cv2.cvtColor(img_localization, cv2.COLOR_BGR2RGB)

                    # --- BƯỚC 6: HIỂN THỊ KẾT QUẢ PHÂN TÁCH TRÊN WEB ---
                    st.success("🎉 Quá trình trích xuất và phân tích hình học hoàn tất!")
                    
                    # Khu vực chỉ số đo lường
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="Độ tương đồng vân tay", value=f"{match_percentage:.2f} %")
                    with m_col2:
                        st.metric(label="Số cặp điểm trùng khớp được đánh số", value=f"{len(good_matches)} điểm")

                    # PHẦN 1: BẢN ĐỒ ĐỐI CHIẾU ĐIỂM ĐẶC TRƯNG (Ở TRÊN)
                    st.markdown("### 📊 1. Sơ Đồ Đối Chiếu Các Điểm Đặc Trưng Đồng Nhất")
                    st.caption("Cách tra cứu: So sánh các cặp số giống nhau giữa ô bên trái (Màu đỏ) và ô bên phải (Màu xanh dương) để kiểm tra cấu trúc.")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.image(out_img1_rgb, caption="Ảnh 1: Vùng đặc trưng trích xuất (Chỉ đánh dấu ĐỎ)", use_container_width=True)
                    with res_col2:
                        st.image(out_img2_rgb, caption="Ảnh 2: Điểm đối chiếu tương ứng (Chỉ đánh dấu XANH DƯƠNG)", use_container_width=True)

                    st.markdown("---")

                    # PHẦN 2: ẢNH KHOANH VÙNG VỊ TRÍ ĐỘC LẬP (TÁCH XUỐNG DƯỚI)
                    st.markdown("### 🗺️ 2. Bản Đồ Xác Định Vùng Vị Trí Cấu Trúc (Localization Result)")
                    if localization_success and match_percentage >= 40:
                        st.info("🎯 Khung hình chữ nhật màu Xanh Neon dưới đây thể hiện chính xác vị trí và góc nghiêng của vùng vân tay Ảnh 1 khi nằm trên Bản đầy đủ (Ảnh 2).")
                        st.image(img_localization_rgb, caption="Vị trí vùng vân tay được định vị trên ảnh bối cảnh đầy đủ", use_container_width=True)
                    else:
                        st.warning("⚠️ Mật độ điểm hoặc chất lượng đường vân chưa đủ để thuật toán tự động dựng khung bao hình học.")
