# Báo cáo Lab 7: Embedding & Vector Store

**Họ tên:** Chu Bá Tuấn Anh  
**Nhóm:** C401-B1  
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
Khi hai đoạn văn có high cosine similarity, điều đó thường có nghĩa là chúng đang nói về những ý gần nhau hoặc cùng một chủ đề, dù cách diễn đạt có thể khác. Nói ngắn gọn, hai vector của chúng nằm gần cùng một hướng trong không gian embedding.

**Ví dụ HIGH similarity:**
- Sentence A: Python là một ngôn ngữ lập trình bậc cao.
- Sentence B: Python thường được dùng để viết phần mềm và xử lý dữ liệu.
- Tại sao tương đồng: Cả hai câu đều nói về Python trong ngữ cảnh lập trình, nên ý nghĩa khá gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: Cơ sở dữ liệu vector dùng để lưu embedding.
- Sentence B: Hôm nay trời mưa khá lớn ở thành phố.
- Tại sao khác: Hai câu gần như không liên quan về chủ đề hay ngữ cảnh.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Với text embeddings, điều quan trọng thường là hướng của vector hơn là độ dài tuyệt đối. Cosine similarity đo mức độ giống nhau về ngữ nghĩa ổn định hơn, trong khi Euclidean distance dễ bị ảnh hưởng bởi độ lớn vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
> *Trình bày phép tính:*  
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`  
> `= ceil((10000 - 50) / (500 - 50))`  
> `= ceil(9950 / 450)`  
> `= ceil(22.11)`  
> *Đáp án:* `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Khi overlap tăng lên 100 thì số chunk tăng thành `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`. Overlap lớn hơn giúp giữ lại ngữ cảnh ở vùng biên giữa hai chunk, giảm nguy cơ một ý quan trọng bị cắt dở.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:  
Dùng regex `(?<=[.!?])(?:\s+)` để tách câu sau các dấu kết thúc như `.`, `!`, `?`, rồi `strip()` lại từng phần để tránh khoảng trắng thừa. Sau khi có danh sách câu, gom theo `max_sentences_per_chunk` để tạo chunk. Edge case xử lý trước là chuỗi rỗng hoặc chỉ có khoảng trắng thì trả về danh sách rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:  
Thuật toán đi theo thứ tự separator ưu tiên: nếu text đã đủ ngắn thì trả về luôn, còn nếu quá dài thì thử tách bằng separator hiện tại rồi gom lại các mảnh sao cho không vượt `chunk_size`. Nếu separator đó không giúp được, hàm sẽ đệ quy xuống separator kế tiếp; đến mức cuối cùng thì fallback về `FixedSizeChunker` không overlap. Base case rõ nhất là khi chuỗi rỗng, khi chiều dài đã nhỏ hơn hoặc bằng `chunk_size`, hoặc khi không còn separator nào để thử nữa.

### EmbeddingStore

**`add_documents` + `search`** — approach:  
Lưu dữ liệu dưới dạng một danh sách record in-memory, mỗi record có `id` nội bộ, `content`, `metadata`, và `embedding`. Khi search, embed câu query rồi tính dot product với embedding của từng record, sau đó sắp xếp giảm dần theo score và lấy `top_k`.

**`search_with_filter` + `delete_document`** — approach:  
Ở `search_with_filter`, filter record theo metadata trước, rồi mới chạy similarity search trên tập đã lọc để tránh lấy nhầm tài liệu ngoài phạm vi cần tìm. Ở `delete_document`, xóa tất cả record có `metadata["doc_id"]` trùng với `doc_id` đầu vào; nếu có Chroma thì cố gắng xóa đồng bộ bên đó, còn nếu không thì vẫn đảm bảo store in-memory đúng.

### KnowledgeBaseAgent

**`answer`** — approach:  
Để agent chạy theo flow rất thẳng: lấy top-k chunk liên quan từ store, ghép chúng thành phần `Context`, rồi chèn câu hỏi vào prompt. Prompt ngắn gọn, đủ để cho LLM biết phải trả lời dựa trên context đã retrieve, và nếu không có context thì sẽ có một dòng fallback rõ ràng để tránh prompt bị rỗng.

### Test Results

```text
$ python -m pytest tests -v
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-8.3.5, pluggy-1.6.0
collected 43 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 43 passed in 0.13s =============================
```

**Số tests pass:** `43 / 43`

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a programming language. | Python is used to build software. | high | -0.0774 | Không |
| 2 | Vector databases store embeddings. | Embedding stores keep vector representations for retrieval. | high | -0.0532 | Không |
| 3 | Dogs are loyal animals. | The stock market closed lower today. | low | -0.0350 | Có |
| 4 | Chunking helps retrieval quality. | Breaking documents into chunks can improve search. | high | -0.1378 | Không |
| 5 | Machine learning models learn from data. | I forgot my umbrella at home. | low | 0.1123 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Điều bất ngờ nhất là cặp số 5: về mặt ngữ nghĩa thì hai câu khá xa nhau, nhưng score lại dương. Điều này nhắc rằng chất lượng của embedding backend quyết định rất nhiều đến retrieval; nếu embedding chỉ mang tính giả lập hoặc không đủ semantic, score nhìn có vẻ “hợp lệ” nhưng lại không phản ánh đúng nghĩa của câu.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
