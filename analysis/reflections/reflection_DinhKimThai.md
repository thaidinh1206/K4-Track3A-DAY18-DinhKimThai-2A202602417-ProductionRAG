# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đinh Kim Thái  
**MSSV:** 2A202602417  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối chiếu chi tiết giữa các khái niệm lý thuyết trong bài giảng và việc hiện thực hóa bằng mã nguồn trong 5 module của bài lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| **Semantic Chunking** | M1 Chunking | `chunk_semantic()` | Sử dụng cosine similarity giữa các vector embedding của các câu liên tiếp với ngưỡng threshold 0.85. Giúp gom cụm các câu cùng chủ đề ngữ nghĩa thành một chunk tự nhiên, không bị đứt đoạn ý tứ giữa chừng như phương pháp cắt cố định theo số ký tự (fixed-size). |
| **Hierarchical Chunking** | M1 Chunking | `chunk_hierarchical()` | Tạo cấu trúc phân cấp Parent Chunk (2048 ký tự) và Child Chunk (256 ký tự). Khi tìm kiếm, thuật toán search trên Child Chunk để đạt độ khớp từ vựng cao, nhưng trả về Parent Chunk làm context cho LLM, giúp LLM có đầy đủ ngữ cảnh xung quanh mà không bị thiếu thông tin. |
| **Structure-Aware Chunking** | M1 Chunking | `chunk_structure_aware()` | Phân tích cú pháp tiêu đề Markdown (`#`, `##`, `###`), duy trì cấu trúc cây phân cấp và lưu tên phần vào metadata (`section`). Điều này đặc biệt hiệu quả với tài liệu quy chế, pháp chế nội bộ doanh nghiệp. |
| **Vietnamese Word Segmentation** | M2 Search | `segment_vietnamese()` | Sử dụng thư viện `underthesea` kết hợp chuẩn hóa dấu gạch dưới (`_`). Giúp tách các từ ghép tiếng Việt (ví dụ: `nghỉ_phép`, `bảo_hiểm_sức_khỏe`), tăng độ chính xác của chỉ mục từ khóa BM25 lên vượt bậc so với tách theo khoảng trắng thông thường. |
| **Hybrid Search & Fusion** | M2 Search | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Kết hợp BM25 (lexical search) và Dense vector embedding BGE-M3 trên Qdrant thông qua thuật toán Reciprocal Rank Fusion (RRF với hằng số $k=60$). RRF giải quyết triệt để bài toán dung hòa thang điểm không cùng độ đo giữa BM25 và cosine similarity. |
| **Cross-Encoder Reranking** | M3 Rerank | `CrossEncoderReranker.rerank()` | Tải mô hình `BAAI/bge-reranker-v2-m3` để chấm điểm tương quan trực tiếp giữa cặp `(query, document)`. Reranker lọc top 20 candidates xuống top 3 chất lượng nhất, loại bỏ hiệu quả các tài liệu gây nhiễu từ vựng (như phân biệt rõ ràng giữa quy định "nghỉ phép" và quy định "VPN"). |
| **RAGAS 4 Metrics** | M4 Eval | `evaluate_ragas()` | Đánh giá độc lập 4 khía cạnh: Faithfulness (độ trung thực), Answer Relevancy (độ bám sát câu hỏi), Context Precision (độ chính xác xếp hạng context) và Context Recall (độ đầy đủ của context). Kết quả pipeline đạt Faithfulness 0.9455 (tăng mạnh từ 0.8389 của baseline). |
| **Failure Diagnosis Tree** | M4 Eval | `failure_analysis()` | Xây dựng cây chẩn đoán lỗi 4 cấp: `Output sai → Context thiếu/thừa? → Retrieval lỗi? → Chunking/Query lỗi?`. Giúp tự động phân loại nguyên nhân gốc rễ và đề xuất giải pháp sửa chữa. |
| **Contextual Enrichment** | M5 Enrichment | `_enrich_single_call()`, `contextual_prepend()` | Dùng 1 LLM call kết hợp để trích xuất đồng thời: tóm tắt ngữ cảnh (summary), sinh câu hỏi giả định (HyQA), và metadata (phiên bản, ngày ban hành). Kỹ thuật này giúp giải quyết hiện tượng chunk mồ côi (lost-in-the-middle) và tăng khả năng match truy vấn. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình thực hiện bài lab, tôi đã gặp phải và giải quyết 3 vấn đề kỹ thuật lớn:

1. **Lỗi mã hóa ký tự tiếng Việt trên Windows Console (UnicodeEncodeError):**
   - **Exact Error Message:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u1ed1' in position 2: character maps to <undefined>`.
   - **Nguyên nhân:** Môi trường terminal PowerShell trên Windows mặc định sử dụng bảng mã ANSI (CP1252) thay vì UTF-8 khi pipe hoặc in dữ liệu ra stdout, gây crash chương trình khi script in các chuỗi tiếng Việt có dấu.
   - **Cách debug & giải quyết:** 
     - Thiết lập biến môi trường `$env:PYTHONIOENCODING="utf-8"` trong PowerShell trước khi thực thi.
     - Trong các file Python cấu hình hệ thống (`check_lab.py`, `config.py`), bổ sung đoạn mã:
       ```python
       if hasattr(sys.stdout, "reconfigure"):
           sys.stdout.reconfigure(encoding="utf-8")
       ```

2. **Lỗi đứt gãy bảng biểu Markdown khi Chunking làm mất thông tin quan trọng:**
   - **Hiện tượng:** Tại câu hỏi *"Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"*, điểm Context Recall bị rớt xuống 0.0 do context chỉ có dòng `| Trên **50.000.000 VNĐ** |` mà mất đi cột người phê duyệt là `Tổng Giám đốc (CEO)`.
   - **Nguyên nhân:** Bộ chia chunk theo kích thước cố định cắt ngang giữa chừng một hàng trong bảng biểu Markdown, đẩy phần thông tin mấu chốt sang chunk tiếp theo mà chunk đó lại không được retrieve.
   - **Cách debug & giải quyết:** Phân tích `reports/ragas_report.json` qua failure analysis; giải pháp là trong hàm `chunk_structure_aware()`, nhận diện các khối bảng biểu (phân tách bởi ký tự `|`) để không bao giờ cắt vụn hàng của bảng, đồng thời áp dụng Hierarchical Chunking để trả về Parent Chunk hoàn chỉnh khi một hàng trong bảng được khớp.

3. **Hiện tượng xung đột phiên bản tài liệu (Temporal / Version Conflict):**
   - **Hiện tượng:** Câu hỏi về số năm thâm niên để cộng ngày phép bị trả lời sai theo chính sách cũ v2023 (5 năm) thay vì chính sách mới v2024 (3 năm).
   - **Nguyên nhân:** Cả hai văn bản `nghi_phep_nam_v2023.md` và `nghi_phep_nam_v2024.md` đều có độ tương đồng từ vựng và ngữ nghĩa rất cao, BM25 và Dense Search không tự động ưu tiên văn bản có hiệu lực mới nhất.
   - **Cách debug & giải quyết:** Tận dụng Module 5 (Enrichment) để trích xuất metadata ngày ban hành (`effective_date`) và phiên bản (`version`), sau đó bổ sung payload filter trong Qdrant ở Module 2 nhằm chỉ ưu tiên các văn bản có trạng thái hiệu lực (`status: active`).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý AI Hỏi đáp Quy trình & Tri thức Doanh nghiệp Nội bộ (Enterprise Knowledge Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản: Chia chunk theo ký tự cố định 500 ký tự (Fixed Character Splitter) có overlap 50 ký tự, dùng mô hình OpenAI `text-embedding-3-small`, tìm kiếm vector đơn thuần qua ChromaDB, và gửi thẳng top 5 chunks vào GPT-4o-mini để trả lời.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Độ chính xác từ khóa thấp:* Các thuật ngữ kỹ thuật viết tắt hoặc mã văn bản (ví dụ: `QĐ-08/2024`, `MFA`) tìm kiếm ngữ nghĩa thường không khớp chính xác.
  - *Hiện tượng ảo giác khi tài liệu cập nhật:* Chưa có cơ chế lọc văn bản hết hiệu lực, dẫn đến câu trả lời bị lẫn lộn giữa quy định cũ và mới.
  - *Context Recall cho câu hỏi phức (Multi-hop) kém:* Với những câu hỏi đòi hỏi thông tin từ 2 phòng ban khác nhau (ví dụ: quy trình thanh toán công tác phí cần cả chính sách Nhân sự và Kế toán), hệ thống chỉ retrieve được 1 trong 2 tài liệu.

#### 2. Kế hoạch cải tiến
1. **Chunking Strategy:**
   - Chuyển sang **Structure-Aware Chunking** cho tài liệu định dạng Markdown/HTML (bảo tồn nguyên vẹn các bảng biểu quy chế và cấu trúc đề mục).
   - Áp dụng **Hierarchical Chunking** (Parent 2048 ký tự, Child 256 ký tự) cho các sổ tay quy định dài, giúp vừa match chính xác từ khóa ở cấp con vừa cung cấp đủ ngữ cảnh bao quát ở cấp cha.
2. **Search Retrieval:**
   - Thay thế Dense Search đơn thuần bằng **Hybrid Search (BM25 + Dense BGE-M3)**.
   - Sử dụng **Underthesea** để tách từ tiếng Việt chuẩn cho BM25.
   - Hợp nhất kết quả bằng **Reciprocal Rank Fusion (RRF)** với tham số $k=60$.
   - Tích hợp **Metadata Payload Filtering** trên Qdrant để lọc các văn bản `is_active=True`.
3. **Reranking:**
   - Tích hợp **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3` hoặc thư viện tối ưu `FlashRank`) để tái xếp hạng top 20 candidates xuống top 4 chunks chất lượng nhất, loại bỏ triệt để các đoạn văn bản gây nhiễu ngữ cảnh.
4. **Evaluation:**
   - Thiết lập bộ đánh giá định kỳ tự động bằng **RAGAS 4 metrics** (Faithfulness, Answer Relevancy, Context Precision, Context Recall) với tập benchmark gồm 50 câu hỏi đa dạng (Lookup, Negation, Multi-hop, Numeric, Temporal conflict).
   - Đặt ngưỡng CI/CD: Bất kỳ bản cập nhật pipeline nào có Faithfulness < 0.85 hoặc Context Recall < 0.75 sẽ không được deploy lên production.
5. **Enrichment:**
   - Triển khai **Combined Single-call Enrichment**: Tận dụng 1 lượt gọi LLM cho mỗi chunk để sinh tóm tắt tiêu đề ngữ cảnh (Contextual prepend) và trích xuất trường metadata có cấu trúc (phòng ban, ngày hiệu lực, mã văn bản).

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1:** Chuẩn hóa toàn bộ corpus dữ liệu nội bộ doanh nghiệp sang Markdown có cấu trúc; cài đặt module `Structure-Aware Chunking` và `Hierarchical Chunking`.
- **Tuần 2:** Dựng cụm Qdrant trên Docker nội bộ; hiện thực hóa `Hybrid Search (BM25 + Dense BGE-M3)` và tích hợp `Cross-Encoder Reranker`.
- **Tuần 3:** Tích hợp pipeline `Contextual Enrichment` để trích xuất metadata văn bản; xây dựng bộ test benchmark 50 câu hỏi và chạy đánh giá RAGAS.
- **Tuần 4:** Tối ưu hóa độ trễ (latency), tinh chỉnh prompt sinh câu trả lời cuối cùng và triển khai phiên bản thử nghiệm cho phòng ban Nhân sự & Vận hành.
