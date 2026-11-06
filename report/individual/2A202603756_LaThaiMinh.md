# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603756
- **Họ và tên:** La Thái Minh
- **SEED:** 598

## 1. Bằng chứng

| Tệp | Tình trạng |
| --- | --- |
| [day01_lab.ipynb](./day01_lab.ipynb) — notebook có code, câu trả lời và output | Có |
| [lab1_report.json](./lab1_report.json) — sinh từ ô báo cáo cuối | Có; **15/15** kiểm tra đạt, **10/10** dự đoán đã ghi |
| `hand_calc.jpg` — ảnh tính tay theo yêu cầu nộp bài | **Chưa có**; hiện chỉ có [hand_calc.pdf](./hand_calc.pdf), cần xuất JPG thấy rõ các bước |

Năm giá trị trong `HAND_CALC` đều qua kiểm tra. Ảnh tính tay vẫn là bằng chứng cần bổ sung riêng.

## 2. Kết quả chính

Số liệu lấy từ [lab1_report.json](./lab1_report.json) và bản [notebook đã chạy](./day01_lab.ipynb). F1 của LLM chỉ đo trên **SUB 40 tin**; F1 test của năm mô hình đầu đo trên toàn bộ tập test. Với SUB nhỏ, chênh lệch F1 vài phần trăm chưa đủ để xếp hạng chắc chắn.

| Phương án | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 (ms/tin) | USD/1.000 tin trong notebook |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0,912 | 0,788 | 8,9 | 14.438 | 0,316 | 0 API |
| BoW + MLP | 0,940 | 0,824 | 27,7 | 29.803 | 0,539 | 0 API |
| MiniLM + kNN | 0,860 | 0,865 | 118,0 | 34,5 | 65,759 | 0 API |
| MiniLM + LogReg | 0,937 | 0,947 | 118,5 | 44,6 | 59,715 | 0 API |
| MiniLM + MLP head (3c) | **0,947** | **0,947** | 119,5 | 43,6 | **45,205** | 0 API |
| Groq GPT-OSS 20B zero-shot | — | 0,927 | 0 | 0,94 | 2.700,545 | Chưa tính* |
| Groq GPT-OSS 120B zero-shot | — | 0,900 | 0 | 1,08 | 2.523,316 | Chưa tính* |
| Groq GPT-OSS 20B few-shot | — | 0,900 | 0 | 0,65 | 3.450,229 | Chưa tính* |

\* Notebook in `$0.00000` cho LLM vì bảng giá trong code chưa có hai model Groq; đó **không phải chi phí thực bằng 0**. Cột 0 API của mô hình cục bộ cũng chưa gồm chi phí máy chủ, điện và bảo trì. Các thời gian đo phụ thuộc môi trường Colab.

## 3. Dự đoán và thực tế

1. **`llm_models`:** Tôi dự đoán GPT-OSS 120B đạt F1 khoảng 0,90, cao hơn 20B khoảng 0,85. Kết quả SUB 40 là **20B 0,927** và **120B 0,900**. Model lớn hơn không mặc nhiên thắng trong mẫu nhỏ; chỉ vài tin sai đã làm F1 thay đổi rõ.
2. **`llm_fewshot`:** Tôi dự đoán thêm bốn ví dụ có thể đưa F1 20B từ 0,927 lên khoảng 0,94 hoặc giữ gần như cũ. Kết quả còn **0,900**; p95 tăng từ **2.700,545** lên **3.450,229 ms/tin**. Các ví dụ được chọn có thể không giúp nhận diện những tin trong SUB, còn prompt dài làm tăng thời gian xử lý.
3. **`hard_eval`:** Tôi dự đoán BoW + MLP đạt khoảng **64% accuracy** trên 50 case khó, thực tế **52%**. Tôi cũng đoán Embedding + MLP head khoảng **74%**, thực tế **60%**. Tập khó cố ý chứa kiểu viết lạ và ngữ cảnh dễ gây nhầm, nên điểm trên test thường quá lạc quan về độ bền của mô hình.

## 4. Phân tích 50 case khó

Trên **50 case gốc**, BoW + MLP tụt F1 nhiều nhất: từ **0,824** trên SUB xuống **0,333** trên tập khó, giảm **0,491**. BoW + LogReg giảm từ 0,788 xuống 0,378. Với BoW + MLP, hai nhóm *spam giọng tự nhiên* và *né từ khóa* đều có tỷ lệ đúng **0%** theo bảng nhóm. BoW chỉ nhìn các token có trong từ điển, nên dễ bỏ lọt quảng cáo viết như hội thoại hoặc cố tình biến dạng chữ.

- **Case #11 — spam giọng tự nhiên:** Tin giả lời người quen ở phòng gym, rồi đưa ưu đãi hội viên và yêu cầu trả lời `JOIN`. BoW + MLP đoán sai vì lời chào thân mật không giống nhiều mẫu spam có từ khóa rõ ràng.
- **Case #29 — né từ khóa:** Tin thưởng giả viết `C0ngr4tulat10ns`, `w0n`, `g1ft c4rd` bằng cách thay chữ bằng số. BoW + MLP đoán sai vì token biến dạng không khớp từ trong vocab. Có thể thử chuẩn hóa đầu vào, đồng thời đo nguy cơ chặn nhầm ham.

LLM nhỏ zero-shot đạt F1 **0,875** trên 50 case khó; prompt phòng thủ đạt **0,935**, nhưng vẫn chặn nhầm bốn tin ham, gồm case #2 về pizza miễn phí ở văn phòng. Tập 50 case viết tay nhằm tìm lỗi, không đại diện cho lưu lượng SMS thực tế. Sáu `MY_CASES` đã được thêm **sau** lần chạy stress test lưu trong notebook, nên các số trên **không** bao gồm chúng.

## 5. Thử thách cá nhân

**Đề #5:** Viết bốn SMS gồm hai spam né từ khóa và hai ham chứa `free` hoặc `win`, rồi so BoW + LogReg với Embedding + LogReg. Code dùng lại `vectorize`, `predict_proba`, MiniLM và các trọng số đã huấn luyện; ngưỡng phân loại là 0,5. **Code và bốn tin có AI hỗ trợ**, được ghi rõ ở mục 7; chúng không đáp ứng điều kiện tự viết, không dùng AI của đề.

| Tin | Nhãn thật | BoW: p(spam), dự đoán | Embedding + LogReg: p(spam), dự đoán | Nhận xét |
| --- | --- | --- | --- | --- |
| 1. Ưu đãi thành viên câu lạc bộ, yêu cầu trả lời `JOIN` | spam | 0,275 → ham ✗ | 0,990 → spam ✓ | Quảng cáo viết như lời trò chuyện; embedding bắt được ý chào mời trong case này. |
| 2. Yêu cầu gửi thông tin thẻ để giữ quyền lợi tài khoản | spam | 0,392 → ham ✗ | 0,172 → ham ✗ | Cả hai bỏ lọt; xác suất riêng lẻ không chứng minh chắc đặc trưng nào gây lỗi. |
| 3. Hỏi bạn có `free` sau giờ học để làm bài nhóm | ham | 0,044 → ham ✓ | 0,252 → ham ✓ | `Free` nghĩa là rảnh, không phải khuyến mãi. |
| 4. Kể đội bóng vừa có trận `win` | ham | 0,040 → ham ✓ | 0,001 → ham ✓ | `Win` là thắng trận, không phải trúng thưởng. |

BoW đúng **2/4**, Embedding + LogReg đúng **3/4**. Bốn mẫu do AI gợi ý và được chọn có chủ đích; không dùng tỷ lệ này để suy ra chất lượng triển khai. Case 2 cho thấy cần thêm dữ liệu lừa đảo tương tự để đánh giá và cải thiện.

## 6. Phương án cho sản phẩm thật

Nhà mạng cần xử lý **10 triệu SMS/ngày**, khoảng **116 tin/giây trung bình**, với p95 < 50 ms/tin. Theo phép đo hiện tại, BoW + LogReg (**0,316 ms**), BoW + MLP (**0,539 ms**) và MiniLM + MLP head 3c (**45,205 ms**) đạt ngưỡng p95. Trong ba cách, 3c có F1 test cao nhất (**0,947**), nhưng thông lượng chỉ **43,6 tin/giây**. Nếu dùng 3c ở quy mô thật, phải chạy song song nhiều phiên xử lý, dự phòng giờ cao điểm và đo lại dưới tải. LLM có p95 trên **2,5 giây**, không thích hợp làm bước đồng bộ cho mọi tin.

Tôi đề xuất **router hai tầng**: BoW + LogReg xử lý ngay các tin có xác suất rất thấp hoặc rất cao; tin lưỡng lự, ví dụ `p(spam) ∈ [0,3; 0,7]`, chuyển sang 3c. Tin mà hai mô hình bất đồng hoặc có dấu hiệu lừa đảo nghiêm trọng vào hàng kiểm tra riêng; LLM chỉ hỗ trợ nhánh này nếu không làm đường xử lý chính chờ API. Cần đo F1, tỷ lệ chặn nhầm ham, p95 và thông lượng **của cả router** dưới tải thật; không thể cộng trực tiếp các p95 riêng lẻ để bảo đảm SLA.

Để ước tính tiền API, giả sử **100 token đầu vào, 1 token đầu ra mỗi SMS**, không tính token suy luận hoặc cache, trong 30 ngày có 300 triệu SMS. Theo giá niêm yết của Groq, [GPT-OSS 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b) giá $0,075/1 triệu token vào và $0,30/1 triệu token ra, khoảng **$2.340/tháng**; [GPT-OSS 120B](https://console.groq.com/docs/model/openai/gpt-oss-120b) giá $0,15 và $0,60, khoảng **$4.680/tháng**. Đây là kịch bản minh họa; số token thực, giá và hạ tầng có thể làm chi phí khác đi.

## 7. AI log và điều đã học

Bảng sau ghi hỗ trợ AI trong cuộc trò chuyện này. Phần thử thách có AI hỗ trợ dù luật đề yêu cầu không dùng AI; tôi không ghi code hoặc bốn SMS là tự làm. Nếu dùng AI ở nơi khác, cần bổ sung trước khi nộp.

| Phần | Tôi hỏi AI gì | AI hỗ trợ gì | Tôi đã kiểm tra hoặc còn cần kiểm tra gì |
| --- | --- | --- | --- |
| Hướng dẫn lab | Đọc dự án, viết hướng dẫn từng bước. | Soạn tài liệu hướng dẫn và giải thích các phần. | Đối chiếu trình tự các ô khi chạy trên Colab. |
| Phần 4 LLM | Hỏi lỗi 4A, dự đoán Groq, few-shot và Câu hỏi 4. | Gợi ý code, dự đoán và lời giải thích. | Bộ kiểm tra 4.1–4.4 đạt; đối chiếu F1 zero-shot 20B 0,927, 120B 0,900, few-shot 0,900. |
| Bắt lỗi, đánh đổi | Hỏi Bắt lỗi 3, `tradeoff` và Câu hỏi 5. | Giải thích lỗi và cách đọc biểu đồ Pareto. | Đối chiếu bảng SUB và ba cách đạt p95 < 50 ms trong bản chạy nộp kèm. |
| Stress test | Hỏi `hard_eval`, Câu hỏi 6 và sáu `MY_CASES`. | Gợi ý dự đoán, phân tích lỗi và sáu case. | Đối chiếu kết quả **50 case gốc**; sáu case thêm chưa được đánh giá trong bản nộp kèm. |
| Tính tay | Hỏi tạo `hand_calc.md`. | Gợi ý năm phép tính. | `HAND_CALC` đạt 5/5; cần ảnh các bước viết tay. |
| Thử thách #5 | Hỏi code, bốn SMS và cách giải thích output. | Viết code so sánh, bốn SMS và đoạn phân tích. | Chạy code: BoW đúng 2/4, Embedding + LogReg đúng 3/4; ghi rõ nguồn AI. |
| Phản ánh, báo cáo | Hỏi phản ánh cuối bài, AI log và hoàn thiện báo cáo. | Gợi ý router, ước tính chi phí, biên tập báo cáo. | Đối chiếu số với `lab1_report.json`; chi phí Groq dựa trên giả định token. |

**Điều tôi hiểu thêm:** F1 cao trên test thường không bảo đảm độ bền với SMS viết khác phân phối; stress test làm F1 BoW + MLP giảm từ 0,824 xuống 0,333. Embedding và LLM xử lý được một số ngữ cảnh khó hơn nhưng vẫn có lỗi riêng, như tin yêu cầu thông tin thẻ mà cả BoW và Embedding bỏ lọt. Chọn mô hình cho sản phẩm cần xét đồng thời chất lượng, độ trễ, thông lượng và loại lỗi, rồi đo lại toàn hệ thống dưới tải thật.
