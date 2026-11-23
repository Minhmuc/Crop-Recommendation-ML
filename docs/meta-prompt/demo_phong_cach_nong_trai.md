# Meta-prompt: Làm lại giao diện demo theo phong cách nông trại pixel

> Lập ngày 23/11/2026. Đây là bản yêu cầu chuẩn cho việc làm lại giao diện `app.py`. Có thể giao cho trợ lý AI hoặc để một thành viên tự làm.

## Vai trò

Bạn là lập trình viên Streamlit kiêm người thiết kế giao diện game. Nhiệm vụ là làm lại giao diện ứng dụng minh họa KNN của nhóm 08 theo phong cách **game nông trại pixel ấm cúng**, cảm hứng từ thể loại game nông trại như Stardew Valley. Đồng thời, bạn bỏ những thông tin thừa trên giao diện mà không làm mất nội dung học thuật cần cho vấn đáp.

## Bối cảnh đã xác minh

| Thành phần | Vị trí | Trạng thái |
|---|---|---|
| Giao diện hiện tại | `app.py` | Có sidebar cấu hình, 3 tab, bảng vị trí trong khoảng train, thẻ cho cả cây có điểm 0 |
| Phần xử lý | `demo_core.py` | Nạp `outputs/crop_{a,b}_02`, kiểm tra đầu vào, top-3, tìm láng giềng, sinh mẫu ngẫu nhiên; đã kiểm thử |
| Giao diện màu | `.streamlit/config.toml` | Theme xanh lá, chưa có font riêng |
| Streamlit | `.venv` | 1.67.0 |

## Quyết định của người dùng (23/11/2026)

1. **Mức phong cách: đậm, kiểu game.**
   - Font pixel cho tiêu đề.
   - Khung gỗ viền nổi, nền giấy da / cỏ / đất.
   - Nút bấm kiểu game; thẻ cây trồng như ô vật phẩm.
   - Được phép dùng CSS.
2. **Bỏ bốn phần thừa:**
   - Sidebar cấu hình KNN và đường dẫn file. Chuyển chọn bộ A/B lên đầu trang; cấu hình KNN chuyển vào tab sổ tay.
   - Bảng "vị trí trong khoảng train" có thanh tiến độ. Đã có cảnh báo khi giá trị nằm ngoài khoảng train nên không cần.
   - Thẻ cây có điểm 0 và ghi chú lặp về điểm 0. Chỉ hiện cây có phiếu.
   - Chú thích kỹ thuật:
     - dòng "Nguồn dữ liệu nhập";
     - chú thích cách tính khoảng cách dưới bảng láng giềng;
     - tên tiếng Anh trên thẻ.
3. **Giọng văn: game + thuật ngữ.** Tiêu đề mang màu sắc game ("Gợi ý mùa vụ", "Hàng xóm gần nhất"). Vẫn giữ các thuật ngữ KNN, điểm phiếu, láng giềng, F1 macro, kiểm định chéo để dùng khi vấn đáp.

## Đầu ra cần tạo

1. **`.streamlit/config.toml`**
   - Theme sáng, khóa chế độ.
   - Bảng màu gỗ / giấy da / cỏ.
   - Font hỗ trợ tiếng Việt: chữ thân **Nunito**, tiêu đề pixel **VT323** (Google Fonts, có bộ ký tự tiếng Việt).
   - Màu chính đủ tối để chữ trắng trên nút dễ đọc.
2. **`app.py`** — bố cục mới:
   - **Băng rôn đầu trang:** bầu trời, mặt trời pixel, đồi cỏ, hàng rào vẽ bằng CSS; tiêu đề font pixel.
   - **Chọn bộ dữ liệu** bằng `st.segmented_control` (`required=True`), kèm số mẫu và số loại cây. Khi đổi bộ, xóa nhãn thật của mẫu test cũ.
   - **Tab "Gợi ý mùa vụ":**
     - Khung gỗ chứa 3 nút sinh mẫu (mẫu thật từ test, ngẫu nhiên trong khoảng train, mẫu ví dụ) và 7 ô nhập. Tooltip của mỗi ô vẫn nêu khoảng train.
     - Ô vật phẩm cho từng cây có phiếu (tối đa 3): biểu tượng lớn, tên tiếng Việt, hạng, thanh năng lượng chia đoạn thể hiện điểm phiếu.
     - Hộp thoại kiểu game: báo đúng/sai khi mẫu lấy từ tập test; cảnh báo khi giá trị ngoài khoảng train hoặc có nhãn đồng điểm.
     - Bảng "Hàng xóm gần nhất": cây trồng, khoảng cách, 7 chỉ số; ẩn `sample_id`.
   - **Tab "Sổ tay nông trại":**
     - Cấu hình KNN_final.
     - Chỉ số F1 macro / Accuracy test của KNN.
     - Bảng so sánh với Gaussian NB, Decision Tree, Random Forest.
     - Đoạn ngắn về cách đọc điểm phiếu và giới hạn dữ liệu.
3. **`demo_core.py`:** chỉ chỉnh tối thiểu.
   - Bỏ ghi chú về nhãn điểm 0, vì giao diện không còn hiện các nhãn này.
   - Thêm hàm trả tên cây ngắn (biểu tượng + tiếng Việt).
   - Không đổi logic dự đoán, tìm láng giềng và kiểm tra đầu vào.
4. **README.md**: cập nhật mục 12 cho khớp giao diện mới.

## Ràng buộc bắt buộc

- **Bản quyền:**
  - Không dùng tên "Stardew Valley", logo, nhân vật, sprite, nhạc, font hay hình ảnh của game gốc.
  - Mọi hình vẽ là CSS / emoji tự thiết kế.
  - Không ghi trên giao diện rằng đây là bản sao của game nào.
- Không huấn luyện lại mô hình, không đổi outputs, không đổi quy tắc kiểm tra đầu vào (vẫn khớp notebook mục 16).
- Số liệu trên giao diện đọc trực tiếp từ `comparison.csv` và gói joblib, không gõ tay.
- **Theo hướng dẫn Streamlit 1.67:**
  - Không dùng `use_container_width`.
  - CSS nhắm vào lớp `.st-key-<key>` của container / nút có `key`, không nhắm vào lớp nội bộ của Streamlit khi có thể tránh.
  - Có `alt=` cho bảng.
- Điểm phiếu luôn được gọi đúng tên: không gọi là "xác suất thành công".
- Không commit Git.

## Điều chỉnh lần 2 (23/11/2026): bớt chất game

Người dùng thấy bản pixel quá giống game và yêu cầu tham khảo các hệ thống khác. Quyết định mới **thay thế mục "Mức phong cách" và "Giọng văn" ở trên**:

- **Hướng thiết kế: sổ tay nông trại ấm**, như ứng dụng lập kế hoạch vườn / farm journal.
  - Nền kem, màu đất – cỏ – gỗ dịu, khung bo góc có bóng nhẹ.
  - Chất nông trại chỉ nằm ở bảng màu.
  - Font: Lora cho tiêu đề, Nunito cho chữ thân.
- **Chỉ giữ emoji cây trồng.** Bỏ băng rôn cảnh nông trại, font pixel, khung gỗ viền nổi, nút kiểu game, thanh điểm chia nấc và hộp thoại game.
  - Dùng thành phần gốc của Streamlit: `st.container(border=True)`, `st.progress`, `st.badge`, `st.success` / `st.error` / `st.warning`.
  - Icon dùng Material Symbols.
- **Tên mục trung tính:** "Chỉ số đất và khí hậu", "Cây trồng được gợi ý", "Láng giềng gần nhất trong tập train", "Thông tin mô hình".
- Các phần thừa đã bỏ ở lần 1 vẫn giữ nguyên là bỏ.

## Kiểm tra trước khi bàn giao

- [ ] `py_compile` qua cho `app.py` và `demo_core.py`.
- [ ] Chạy app headless với agent API:
  - tải trang không lỗi;
  - bấm đủ 3 nút sinh mẫu trên cả A và B;
  - đổi bộ dữ liệu thì nhãn thật cũ biến mất;
  - nhập giá trị ngoài khoảng train thì hiện cảnh báo;
  - console không có cảnh báo hoặc deprecation mới.
- [ ] Mẫu ví dụ trên B vẫn cho Lúa (rice) với điểm phiếu 100%.
- [ ] Báo lại cho người dùng phần cần tự xem bằng trình duyệt: màu sắc, font, khung gỗ. Agent API không kiểm tra được phần hiển thị.
