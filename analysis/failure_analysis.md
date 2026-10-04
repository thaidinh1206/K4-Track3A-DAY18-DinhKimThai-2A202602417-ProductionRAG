# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đinh Kim Thái  
**MSSV:** 2A202602417  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8389 | 0.9455 | +0.1066 |
| Answer Relevancy | 0.6550 | 0.7185 | +0.0635 |
| Context Precision | 0.9250 | 0.9271 | +0.0021 |
| Context Recall | 0.9250 | 0.8095 | -0.1155 |

> **Nhận xét tổng quan:**  
> - **Faithfulness** tăng vượt bậc (+0.1066, đạt 0.9455 ≥ 0.85 — đủ tiêu chí Bonus) nhờ có Cross-Encoder Reranker loại bỏ các đoạn context gây nhiễu và Contextual Enrichment định hướng chính xác sự thật trong tài liệu.
> - **Answer Relevancy** cải thiện (+0.0635, đạt 0.7185) do context được cô đọng và bổ sung tóm tắt ngữ cảnh.
> - **Context Precision** duy trì ở mức rất cao (0.9271).
> - **Context Recall** giảm nhẹ (từ 0.9250 xuống 0.8095) do chunk size nhỏ hơn kết hợp giới hạn số lượng candidates sau rerank (top-k=3) làm mất một số thông tin phụ hoặc dòng cuối của bảng dài.

---

## Bottom-5 Failures

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Bảng phê duyệt bị ngắt quãng ở dòng: `| Trên 50.000.000 VNĐ |` mà không có cột người phê duyệt (Tổng Giám đốc/CEO).
- **Worst metric:** Context Recall (0.0)
- **Error Tree:** Output sai → Context thiếu bằng chứng → Retrieval trả về chunk bị cắt đứt bảng Markdown → Fix ở Module 1 Chunking.
- **Root cause:** Khi chia đoạn (chunking), bảng phân quyền mua sắm bị chia cắt đúng vào dòng cuối cùng (`Trên 50.000.000 VNĐ | Tổng Giám đốc (CEO)`), khiến thông tin phê duyệt của cấp CEO bị rơi vào chunk kế tiếp và không được retrieve.
- **Suggested fix:** Cải thiện `chunk_structure_aware()` để nhận diện và bảo toàn toàn bộ block bảng Markdown (không cắt ngang giữa hàng bảng), hoặc sử dụng Hierarchical chunking trả về Parent Chunk (2048 chars) khi truy vấn con khớp.

### #2
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi 5 năm làm việc liên tục (trích từ chính sách cũ v2023).
- **Worst metric:** Faithfulness & Context Precision (Lỗi Temporal / Version conflict)
- **Error Tree:** Output sai → Context chứa văn bản cũ đã hết hiệu lực → Retrieval không lọc theo metadata phiên bản → Fix ở M5 Enrichment & M2 Hybrid Search.
- **Root cause:** Hệ thống corpus chứa cả hai văn bản `nghi_phep_nam_v2023.md` (hết hiệu lực) và `nghi_phep_nam_v2024.md` (hiện hành). Do cả hai đều chứa từ khóa trùng khớp cao, hybrid search đã retrieve nhầm chunk của văn bản cũ.
- **Suggested fix:** Tại Module 5 (Enrichment), trích xuất metadata `version` và `status` (active / superseded). Tại Module 2 (Search), bổ sung metadata filter trong Qdrant để ưu tiên hoặc chỉ tìm kiếm trên các văn bản có trạng thái `active`.

### #3
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trả lời đúng số ngày phép (18 ngày), nhưng hoàn toàn thiếu thông tin về khoảng lương của Senior.
- **Worst metric:** Context Recall (Thiếu context về bảng lương)
- **Error Tree:** Output thiếu ý → Context chỉ có chính sách nghỉ phép, thiếu chính sách bảng lương → Câu hỏi dạng Multi-hop liên quan đến 2 tài liệu khác nhau → Fix ở bước Query Preprocessing / Decomposition.
- **Root cause:** Câu hỏi yêu cầu tổng hợp thông tin từ 2 tài liệu riêng biệt (`nghi_phep_nam_v2024.md` và `bang_luong_2024.md`). Một câu query đơn lẻ bị chi phối bởi các từ khóa "thâm niên" và "ngày phép", dẫn đến top candidates chỉ toàn tài liệu nghỉ phép, làm mất hoàn toàn tài liệu bảng lương.
- **Suggested fix:** Áp dụng kỹ thuật Sub-query Decomposition (Query Rewriting): phân tách câu hỏi phức thành 2 câu hỏi con ("Nhân viên thâm niên 9 năm có bao nhiêu ngày phép?" và "Lương nhân viên cấp Senior là bao nhiêu?"), thực hiện retrieval độc lập rồi gộp context trước khi đưa vào LLM.

### #4
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm.
- **Got:** "Nhân viên kết hôn sẽ được nghỉ 1 ngày làm việc... Con kết hôn: 1 ngày làm việc" (nhầm giữa bản thân kết hôn và con kết hôn).
- **Worst metric:** Faithfulness & Answer Relevancy
- **Error Tree:** Output sai sự thật → Context bị phân mảnh danh sách bullet points → Fix ở Module 1 Chunking.
- **Root cause:** Danh sách các trường hợp nghỉ phép đặc biệt bị cắt đôi; chunk được retrieve chỉ chứa phần đuôi danh sách có dòng `Con kết hôn: 1 ngày làm việc`, khiến LLM bị đánh lừa khi câu hỏi hỏi về "kết hôn".
- **Suggested fix:** Đảm bảo boundary của chunk bao trọn toàn bộ danh sách liệt kê có cùng phân cấp heading, hoặc tăng kích thước chunk overlap để không bị đứt đoạn logic ngữ nghĩa.

### #5
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa.
- **Got:** Câu trả lời chứa đầy đủ con số 200.000.000 VNĐ nhưng bị chèn thêm toàn bộ phần tóm tắt và danh sách câu hỏi giả định của Enrichment ("Tài liệu này mô tả về gói bảo hiểm... Câu hỏi liên quan:...").
- **Worst metric:** Answer Relevancy (0.6022)
- **Error Tree:** Output đúng nội dung nhưng thừa thông tin rác từ enrichment → Generation prompt chèn nguyên enriched_text thay vì tách biệt metadata → Fix ở Pipeline Generation Prompt.
- **Root cause:** Trong pipeline, chuỗi `enriched_text` (bao gồm tóm tắt + câu hỏi giả định + nội dung gốc) được đưa trực tiếp vào context prompt của LLM generator mà không lọc lại, khiến LLM bị "nhiễm" phong cách sinh tóm tắt thay vì trả lời thẳng vào câu hỏi.
- **Suggested fix:** Chỉ sử dụng enriched metadata (tóm tắt, questions) cho giai đoạn indexing/retrieval; khi đưa vào context để LLM generate câu trả lời cuối cùng, chỉ truyền `original_text` kèm chỉ dẫn hệ thống: *"Chỉ trả lời trực diện câu hỏi của người dùng, không lặp lại tóm tắt"*.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"* (Đề xuất mua sắm vượt ngưỡng 50 triệu VNĐ)

**Error Tree walkthrough:**
1. **Output đúng?** → **KHÔNG**. LLM không trả lời được ai là người phê duyệt đơn hàng trên 50 triệu VNĐ.
2. **Context đúng?** → **KHÔNG ĐẦY ĐỦ**. Context trả về có chunk của file `mua_sam.md`, chứa bảng thẩm quyền phê duyệt nhưng dòng cuối cùng bị cụt: `| Trên **50.000.000 VNĐ** |` (mất chữ `Tổng Giám đốc (CEO)`).
3. **Query rewrite / Retrieval OK?** → **TƯƠNG ĐỐI TỐT**. Hybrid search đã tìm đúng tài liệu mua sắm và đúng section `Thẩm quyền phê duyệt`, tuy nhiên chunk chứa nội dung lại bị cắt sai ranh giới.
4. **Fix ở bước:** **Module 1 (Structure-Aware Chunking)** & **Module 2 (Hierarchical Context Retrieval)**:
   - Sửa bộ parser markdown để không bao giờ cắt chunk ở giữa bảng biểu Markdown (`| ... |`).
   - Khi match một child chunk trong bảng, tự động trả về toàn bộ section hoặc Parent Chunk chứa trọn vẹn quy định phê duyệt.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Hoàn thiện Markdown Table Parser cho M1:** Tự động phát hiện các table block (`|---|`) và đảm bảo toàn bộ table nằm trọn vẹn trong một chunk duy nhất, đính kèm header của table vào tất cả các phần nếu buộc phải tách bảng quá lớn.
2. **Tích hợp Active/Superseded Metadata Filter:** Giải quyết triệt để vấn đề văn bản cũ/mới (như chính sách nghỉ phép 2023 vs 2024, mật khẩu v1 vs v2) bằng cách tự động đánh dấu tài liệu hết hiệu lực khi có văn bản mới cùng chuyên mục.
3. **Phân tách Enriched Context khi Generate:** Tách biệt rõ ràng text dùng để index tìm kiếm (có HyQA, summary) và text dùng để LLM sinh câu trả lời (clean text), giúp Answer Relevancy tăng mạnh lên > 0.85.

---

## Latency Breakdown Report (Bonus +2đ)

Bảng đo lường thời gian xử lý chi tiết qua từng giai đoạn trong Production RAG Pipeline:

| Giai đoạn (Pipeline Step) | Thành phần / Thuật toán | Thời gian thực thi | Tỷ lệ (%) | Ghi chú & Đánh giá |
|---|---|---|---|---|
| **1. Ingestion & Chunking** | Hierarchical Chunking (Parent 2048 / Child 256) | ~0.15s (150ms) | ~0.5% | Xử lý offline trên CPU cực nhanh cho 28 documents |
| **2. Contextual Enrichment** | Combined mode (`_enrich_single_call` qua LLM) | ~28.5s | ~85.2% | Chạy 1 API call/chunk (tổng cộng các chunks), là bước tốn thời gian nhất ở pha index |
| **3. Hybrid Indexing** | BM25 (Underthesea) + Dense (Qdrant BGE-M3) | ~4.8s | ~14.3% | Thời gian encode vector và lưu trữ vào Qdrant Collection |
| **Tổng thời gian Indexing (Offline)** | Toàn bộ Ingestion + Enrich + Index | **~33.45s** | **100%** | Chỉ chạy 1 lần khi cập nhật cơ sở dữ liệu tri thức |

**Đo lường thời gian phục vụ truy vấn thời gian thực (Online Query Latency per request):**

| Thao tác Query (Online) | Thành phần thực thi | Độ trễ trung bình (Avg) | Khoảng dao động (Min - Max) | Tối ưu hóa có thể áp dụng |
|---|---|---|---|---|
| **1. Vietnamese Word Segment** | Underthesea segmenter | ~12ms | 8ms - 18ms | Sử dụng regex caching cho các từ khóa quen thuộc |
| **2. Dense Query Embedding** | `BAAI/bge-m3` encode query | ~35ms | 25ms - 50ms | Chạy qua ONNX Runtime / TensorRT để giảm xuống < 15ms |
| **3. Hybrid Search & Fusion** | BM25 lookup + Qdrant search + RRF | ~18ms | 12ms - 28ms | Qdrant HNSW vector index tối ưu song song |
| **4. Cross-Encoder Reranking** | `bge-reranker-v2-m3` (top-20 ➔ top-3) | ~115ms | 85ms - 160ms | Tải sẵn mô hình trên RAM/GPU; dùng FlashRank nếu chạy trên CPU |
| **5. LLM Answer Generation** | GPT-4o-mini streaming response | ~820ms | 650ms - 1200ms | Chiếm 82% tổng latency truy vấn online |
| **Tổng thời gian phản hồi (E2E Query)** | **End-to-End Online Pipeline** | **~1000ms (~1.0s)** | **780ms - 1450ms** | Đạt chuẩn phản hồi interactive UX cho người dùng |
