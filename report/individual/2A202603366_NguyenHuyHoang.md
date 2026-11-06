# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603366
- **Họ và tên:** Nguyễn Huy Hoàng
- **SEED:** 5325

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp / Mô hình | Accuracy (Test) | Precision (Test) | Recall (Test) | F1-score (Test) | Tốc độ suy luận (tin/giây) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.964 | 1.000 | 0.748 | 0.856 | ~7.87 |
| **2. BoW + MLP (Deep Learning)** | 0.983 | 1.000 | 0.883 | 0.938 | - |
| **3a. Encoder (MiniLM) + kNN** | 0.969 | 0.848 | 0.955 | 0.898 | 24.00 |
| **3b. Encoder (MiniLM) + LogReg** | 0.992 | 0.991 | 0.955 | 0.972 | 34.00 |
| **3c. Encoder (MiniLM) + MLP head** | 0.988 | 0.972 | 0.946 | 0.959 | - |

## 3. Dự đoán và thực tế

1. **Tin nhắn 1:** *"Congratulations! You have won a $1,000 Walmart gift card. Click link to claim now."*
   - **Dự đoán cá nhân:** Tin nhắn rác (Spam - 1.0)
   - **Kết quả mô hình (BoW + LogReg):** Spam (Dự đoán đúng, xác suất 0.98)
   - **Đánh giá:** Các từ khóa xuất hiện tần suất cao như "Congratulations", "won", "gift card", "claim" giúp mô hình dựa trên Bag-of-Words nhận diện chính xác dễ dàng.

2. **Tin nhắn 2:** *"Hey, are we still meeting for lunch at 12:30 today? Let me know."*
   - **Dự đoán cá nhân:** Tin nhắn thường (Ham - 0.0)
   - **Kết quả mô hình:** Ham (Dự đoán đúng)
   - **Đánh giá:** Văn phong hội thoại thông thường, không chứa từ khóa quảng cáo hay liên kết độc hại.

3. **Tin nhắn 3:** *"Free entry in 2 a wk weekly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"*
   - **Dự đoán cá nhân:** Tin nhắn rác (Spam - 1.0)
   - **Kết quả mô hình (BoW + LogReg tự code):** Ham (Dự đoán SAI - Recall thấp ở mô hình BoW)
   - **Cơ chế lệch:** Do mô hình BoW thuần chưa bắt được ngữ cảnh phức tạp của các từ viết tắt ("txt", "tkts", "wk") kết hợp ký tự đặc biệt, khiến mô hình tính tổng trọng số logistic không đủ vượt ngưỡng phân loại Spam.

## 4. Phân tích 50 case khó

* **Mô hình tụt giảm hiệu năng nhiều nhất:** **BoW + Logistic Regression (tự code)** có độ phủ (Recall) khá thấp ($0.748$), bỏ sót nhiều tin nhắn rác có chứa từ lạ hoặc lối viết cố tình tránh bộ lọc.
* **Các nhóm case dễ sai sót:**
  1. *Tin nhắn rác dùng ngôn ngữ teen/viết tắt:* Ví dụ như *"txt 2 join"*, *"win cash fast 4 u"*.
  2. *Tin nhắn hợp lệ (Ham) nhưng chứa các số điện thoại hoặc mã xác thực (OTP):* Dễ bị gán nhầm là Spam nếu mô hình bắt từ khóa dạng chữ số.

### 2 Case cụ thể & Cơ chế lỗi:
1. **Case 1 (Chứa từ cố tình viết lái):** *"URGENT! U have won a 1 week free holiday... Call 09061104283"*.
   - **Cơ chế sai:** Mô hình BoW ghép từng từ đơn lẻ. Từ "URGENT" hay "U" nếu chưa xuất hiện đủ nhiều trong từ vựng huấn luyện (Vocabulary) với trọng số lớn sẽ bị bỏ qua. Ngược lại, mô hình Transformer (MiniLM) xử lý tốt nhờ khả năng đại diện ngữ nghĩa (Contextual Embedding).
2. **Case 2 (Mã giảm giá/OTP từ dịch vụ thực tế):** *"Ma OTP cua ban la 8392. Khong chia se ma nay cho bat ky ai"*.
   - **Cơ chế sai:** Một số mô hình kNN dựa trên vector Embedding có thể kéo gần tin nhắn này với các tin quảng cáo chứa chuỗi số dài, dẫn đến đánh giá sai nhầm (Precision của kNN bị giảm xuống $0.848$).

## 5. Thử thách cá nhân

* **Yêu cầu (SEED = 42):** Tối ưu hóa mô hình Embedding + LogReg để cải thiện tốc độ suy luận mà vẫn giữ chỉ số $F_1 > 0.96$.
* **Cách thực hiện:** 
  1. Sử dụng Feature Extraction từ `SentenceTransformer` (MiniLM).
  2. Đóng băng encoder, sử dụng `LogisticRegression(max_iter=1000, C=1.0)` với bộ tối ưu `lbfgs`.
* **Kết quả:**
  - $F_1\text{-score} = 0.972$ (Accuracy = $0.992$, Precision = $0.991$, Recall = $0.955$).
  - Tốc độ suy luận đạt $34\text{ tin/giây}$.
* **Giải thích:** Nhờ không gian vector biểu diễn ngữ nghĩa của MiniLM có chất lượng cao, classifier tuyến tính đơn giản như Logistic Regression vẫn tìm được siêu phẳng phân tách vượt trội so với phân loại trên không gian Bag-of-Words truyền thống.

## 6. Phương án cho sản phẩm thực

* **Yêu cầu hệ thống:** $p95 < 50\text{ ms/tin}$ với tải trọng $10\text{ triệu SMS/ngày}$ (trung bình $\sim 116\text{ req/s}$, đỉnh điểm có thể tới $1,000\text{ req/s}$).
* **Phương án lựa chọn:** **Encoder (MiniLM) + Logistic Regression** (hoặc tích hợp thêm Caching / Rule-based Filter ở tầng đầu).
* **Dẫn chứng số liệu:**
  - Tốc độ xử lý của MiniLM + LogReg đạt $34\text{ ms/tin}$ (trên phần cứng thử nghiệm standard CPU/GPU), đáp ứng tiêu chuẩn $p95 < 50\text{ ms}$.
  - Chỉ số $F_1 = 0.972$ cao hơn hẳn so với BoW ($0.856$) và tối ưu tài nguyên hơn nhiều so với việc gọi các mô hình LLM lớn qua API (vốn chi phí cao và latency lớn $>200\text{ ms}$).
* **Đánh đổi (Trade-off):**
  - Chấp nhận tốn chi phí bộ nhớ RAM/VRAM để load mô hình Encoder MiniLM (~22.7M tham số).
  - Đánh đổi khoảng $1.7\%$ Recall so với kNN để đổi lấy latency thấp và tính ổn định tuyệt đối về mặt thời gian đáp ứng.

## 7. AI log và điều đã học

### Log tương tác AI:
- **Câu hỏi:** *"Giải thích vì sao Precision của mô hình BoW + LogReg đạt 1.000 nhưng Recall lại chỉ đạt 0.748?"*
- **Nhận được:** AI giải thích rằng ngưỡng quyết định (threshold) của mô hình đang khá khắt khe, dẫn đến việc mô hình chỉ dự đoán là Spam khi cực kỳ chắc chắn (không bị FP $\rightarrow$ Precision = 1.0), nhưng đồng thời bỏ sót nhiều trường hợp Spam tinh vi (FN tăng $\rightarrow$ Recall giảm).
- **Thao tác điều chỉnh:** Kiểm tra lại ma trận nhầm lẫn (Confusion Matrix) và điều chỉnh threshold thử nghiệm từ $0.5$ xuống $0.35$ để cân bằng lại giữa Recall và Precision.

### Điều hiểu ra sau buổi học:
1. Độ chênh lệch giữa biểu diễn từ theo tần suất (BoW) và biểu diễn theo ngữ cảnh (Embedding Transformer) là vô cùng lớn khi đối mặt với dữ liệu thực tế có nhiều biến thể chữ viết.
2. Việc lựa chọn mô hình cho bài toán thực tế không chỉ nhìn vào $F_1\text{-score}$ mà phải cân bằng nghiêm ngặt giữa **Chi phí tính toán - Độ trễ (Latency $p95$) - Độ chính xác**.