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

> _Trình bày phép tính:_  
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`  
> `= ceil((10000 - 50) / (500 - 50))`  
> `= ceil(9950 / 450)`  
> `= ceil(22.11)`  
> _Đáp án:_ `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Khi overlap tăng lên 100 thì số chunk tăng thành `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`. Overlap lớn hơn giúp giữ lại ngữ cảnh ở vùng biên giữa hai chunk, giảm nguy cơ một ý quan trọng bị cắt dở.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Triết học Mác - Lênin (Giáo trình đại học)

**Tại sao nhóm chọn domain này?**

> _Viết 2-3 câu:_
> Nhóm chọn domain này vì giáo trình lý luận chính trị có cấu trúc chương mục rõ ràng, đồng thời chứa đựng nhiều định nghĩa, khái niệm trừu tượng với logic lập luận chặt chẽ. Việc áp dụng mô hình RAG trên bộ tài liệu này cho phép kiểm tra khả năng duy trì ngữ cảnh cũng như năng lực của mô hình trong việc trích xuất và kết nối thông tin phức tạp.

### Data Inventory

| #   | Tên tài liệu                     | Nguồn                  | Số ký tự | Metadata đã gán                                      |
| --- | -------------------------------- | ---------------------- | -------- | ---------------------------------------------------- |
| 1   | Giáo trình Triết học Mác - Lênin | Bộ Giáo dục và Đào tạo | ~906.610 | `title`, `author`, `year`, `domain`, `document_type` |
| 2   |                                  |                        |          |                                                      |
| 3   |                                  |                        |          |                                                      |
| 4   |                                  |                        |          |                                                      |
| 5   |                                  |                        |          |                                                      |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị                      | Tại sao hữu ích cho retrieval?                                                                          |
| --------------- | ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `title`         | string | "Giáo trình Triết học Mác - Lênin" | Giúp hệ thống định hướng truy xuất đúng tài liệu gốc khi truy vấn.                                      |
| `domain`        | string | "Triết học Mác - Lênin"            | Phân loại thể loại dữ liệu (luật, truyện hay lý luận) để tinh chỉnh phạm vi tìm kiếm.                   |
| `chapter`       | string | "Chương 1"                         | Rất hữu ích với tài liệu dài, giúp hạn chế khoảng tìm kiếm xuống chương cụ thể nếu xác định được topic. |
| `document_type` | string | "textbook"                         | Dùng để filter khi kho tài liệu trộn lẫn sách tóm tắt, đề thi, và giáo trình chính thức.                |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu           | Strategy                         | Chunk Count | Avg Length | Preserves Context?      |
| ------------------ | -------------------------------- | ----------- | ---------- | ----------------------- |
| `giao_trinh.md`    | FixedSizeChunker (`fixed_size`)  | 1522        | 500.0      | Kém (hay cắt ngang câu) |
| `giao_trinh.md`    | SentenceChunker (`by_sentences`) | 966         | 706.1      | Thường chunk quá dài    |
| `giao_trinh.md`    | RecursiveChunker (`recursive`)   | 2034        | 334.8      | Tốt (giữ khối đoạn văn) |
| `python_intro.txt` | FixedSizeChunker (`fixed_size`)  | 5           | 428.8      | Kém                     |
| `python_intro.txt` | SentenceChunker (`by_sentences`) | 3           | 645.7      | Tốt                     |
| `python_intro.txt` | RecursiveChunker (`recursive`)   | 5           | 387.0      | Khá tốt                 |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**

> _Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?_
> Chiến lược thực hiện phân tách văn bản bằng đệ quy dựa trên một mảng các dấu phân tách ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Đầu tiên nó cố gắng tách văn bản mảng lớn theo các đoạn văn (chứa `\n\n` hoặc `\n`). Nếu một cụm vẫn quá lớn (`> chunk_size`), nó sẽ tiếp tục chia nhỏ cụm đó dựa theo dấu phân tách câu chấm câu, từ đó làm giảm tối đa nguy cơ cắt rách ranh giới ngữ nghĩa.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> _Viết 2-3 câu: domain có pattern gì mà strategy khai thác?_
> Giáo trình Triết học có cấu trúc phân đoạn rõ ràng để diễn giải các khái niệm trừu tượng chồng lên nhau. Việc sử dụng RecursiveChunker khai thác được lợi thế của các khoảng trắng và dấu ngắt đoạn `\n\n`, bảo đảm nguyên vẹn 1 unit giải thích lý luận thay vì bị cắt cứng nhắc bằng cố định chiều dài.

**Code snippet (nếu custom):**

```python
from chunking import RecursiveChunker

# Sử dụng RecursiveChunker tích hợp làm strategy chính
chunker = RecursiveChunker(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=500)
chunks = chunker.chunk(document_text)

# chunks chứa các phân đoạn với metadata đính kèm
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu        | Strategy                       | Chunk Count | Avg Length | Retrieval Quality?                                      |
| --------------- | ------------------------------ | ----------- | ---------- | ------------------------------------------------------- |
| `giao_trinh.md` | best baseline (`by_sentences`) | 966         | 706.1      | Bị nhão ngữ cảnh do lấy quá nhiều thông tin             |
| `giao_trinh.md` | của tôi (`recursive`)          | 2034        | 334.8      | Tối ưu vì giới hạn tập trung quanh 1 khái niệm gọn gàng |

### So Sánh Với Thành Viên Khác

| Thành viên          | Strategy                          | Retrieval Score (/10) | Điểm mạnh                                                                                                            | Điểm yếu                                                                                                                      |
| ------------------- | --------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Tôi Chu Bá Tuấn Anh | RecursiveChunker                  | 8.5                   | Cân bằng tốt giữa ngữ nghĩa và độ dài; giữ được cấu trúc tài liệu (paragraph/sentence); retrieval ổn định            | Phụ thuộc heuristic nên đôi khi split chưa tối ưu; có thể tạo chunk rời rạc; cần tuning chunk size và overlap thêm            |
| Nguyễn Mai Phương   | ParentChildChunker (Small-to-Big) | 8                     | Child nhỏ (319 chars) match chính xác thuật ngữ; parent lớn giữ ngữ cảnh section cho LLM; 4/5 queries relevant top-3 | Parent quá lớn (avg 26K chars) có thể vượt context window LLM; heading regex chỉ hoạt động tốt với giáo trình có format chuẩn |
| Chu Thị Ngọc Huyền  | Sentence Chunking                 | 8                     | Bảo toàn ngữ cảnh logic của lập luận triết học bằng cách tôn trọng ranh giới câu, giúp RAG retrieval cao hơn         | Chunk size nhỏ hơn (422 vs 500 chars) có thể bỏ lỡ context nếu lập luận triết học kéo dài trên nhiều câu                      |
| Nguyễn Thị Tuyết    | RecursiveChunker                  | 8                     | Giữ ngữ cảnh tốt, ít cắt ngang đoạn                                                                                  | Cần tinh chỉnh thêm theo chương                                                                                               |
| Hứa Quang Linh      | AgenticChunker                    | 9                     | Tự phát hiện ranh giới chủ đề bằng embedding; mỗi chunk mang đủ ngữ cảnh 1 khái niệm triết học                       | Chunk lớn (avg ~4K chars) có thể chiếm nhiều context window; chạy chậm hơn (~97s trên 684K chars)                             |
| Lĩnh                | AgenticChunker                    | 9                     | Tự phát hiện ranh giới chủ đề bằng embedding; mỗi chunk mang đủ ngữ cảnh 1 khái niệm triết học                       | Chunk lớn (avg ~4K chars) có thể chiếm nhiều context window; chạy chậm hơn (~97s trên 684K chars)                             |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> _Viết 2-3 câu:_
> `RecursiveChunker` rõ ràng là tốt nhất cho tài liệu lý luận chính trị. Với các đoạn văn dài giải thích định nghĩa liên kết với nhau, việc cố gắng tách chúng ở cấp độ đoạn văn hoặc cấp độ câu (nếu đoạn đó quá dài) giúp tránh mất mát ngữ nghĩa cục bộ \- một nhược điểm chí mạng khiến việc retrieval trả về các câu văn không đầy đủ cấu trúc nếu dùng FixedSize.

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

| Pair | Sentence A                               | Sentence B                                                  | Dự đoán | Actual Score | Đúng? |
| ---- | ---------------------------------------- | ----------------------------------------------------------- | ------- | ------------ | ----- |
| 1    | Python is a programming language.        | Python is used to build software.                           | high    | -0.0774      | Không |
| 2    | Vector databases store embeddings.       | Embedding stores keep vector representations for retrieval. | high    | -0.0532      | Không |
| 3    | Dogs are loyal animals.                  | The stock market closed lower today.                        | low     | -0.0350      | Có    |
| 4    | Chunking helps retrieval quality.        | Breaking documents into chunks can improve search.          | high    | -0.1378      | Không |
| 5    | Machine learning models learn from data. | I forgot my umbrella at home.                               | low     | 0.1123       | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Điều bất ngờ nhất là cặp số 5: về mặt ngữ nghĩa thì hai câu khá xa nhau, nhưng score lại dương. Điều này nhắc rằng chất lượng của embedding backend quyết định rất nhiều đến retrieval; nếu embedding chỉ mang tính giả lập hoặc không đủ semantic, score nhìn có vẻ “hợp lệ” nhưng lại không phản ánh đúng nghĩa của câu.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                                 | Gold Answer                                                                                                                           |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Triết học là gì?                                      | Triết học là hệ thống tri thức lý luận chung nhất về thế giới và vị trí con người trong thế giới đó.                                  |
| 2   | Vấn đề cơ bản của triết học gồm những mặt nào?        | Gồm mặt bản thể luận (vật chất - ý thức cái nào có trước) và mặt nhận thức luận (con người có khả năng nhận thức thế giới hay không). |
| 3   | Vai trò của thực tiễn đối với nhận thức là gì?        | Thực tiễn là cơ sở, động lực, mục đích và tiêu chuẩn kiểm tra chân lý của nhận thức.                                                  |
| 4   | Phép biện chứng duy vật nhấn mạnh điều gì?            | Nhấn mạnh sự vận động, phát triển và mối liên hệ phổ biến của sự vật hiện tượng.                                                      |
| 5   | Sự khác nhau giữa chủ nghĩa duy vật và duy tâm là gì? | Duy vật coi vật chất có trước, quyết định ý thức; duy tâm coi ý thức/tinh thần có trước.                                              |

### Kết Quả Của Tôi

| #   | Query                        | Top-1 Retrieved Chunk (tóm tắt)                    | Score | Relevant? | Agent Answer (tóm tắt)                                                                                                                                                                                                                                                                                                                               |
| --- | ---------------------------- | -------------------------------------------------- | ----- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Triết học là gì?             | Thế giới quan là toàn bộ những quan điểm về thế... | 0.633 | Có        | Theo nội dung được cung cấp trong đoạn trích, triết học là **hạt nhân lý luận của thế giới quan**. Ngoài ra, triết học Mác - Lênin còn được mô tả là mang lại **thế giới quan duy vật biện chứng**, đóng vai trò là **hạt nhân thế giới quan cộng sản**.                                                                                             |
| 2   | Vấn đề cơ bản của triết học? | Như vậy, triết học với tính cách là hạt nhân lý... | 0.779 | Có        | Vấn đề cơ bản của triết học là vấn đề về mối quan hệ giữa vật chất với ý thức. Đây là vấn đề có ý nghĩa nền tảng, là điểm xuất phát để giải quyết tất cả những vấn đề còn lại trong triết học. Ph.Ăngghen cũng từng nhấn mạnh: “Vấn đề cơ bản lớn của mọi triết học, đặc biệt là của triết học hiện đại, là vấn đề quan hệ giữa tư duy với tồn tại.” |
| 3   | Vai trò của thực tiễn?       | Sự hoà quyện giữa tri thức với tình cảm và trải... | 0.697 | Có        | Theo nội dung được cung cấp trong đoạn trích, vai trò của thực tiễn rất quan trọng:                                                                                                                                                                                                                                                                  |
| 4   | Phép biện chứng duy vật?     | _b. Khái niệm phép biện chứng duy vật_...          | 0.829 | Có        | Phép biện chứng duy vật là một khái niệm được đề cập trong [Chunk 1], tuy nhiên nội dung cụ thể của khái niệm này không được trình bày rõ ràng trong các đoạn cung cấp. Do đó, không thể đưa ra định nghĩa đầy đủ về phép biện chứng duy vật chỉ từ những thông tin hiện có.                                                                         |
| 5   | Duy vật vs duy tâm?          | Đó là một đường lối khoa học và đường lối đó ch... | 0.456 | Môt phần  | Theo đoạn văn được cung cấp, lập trường **duy vật** được nhấn mạnh là nền tảng cho một đường lối khoa học đúng đắn (xem Chunk 1). Cụ thể:                                                                                                                                                                                                            |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5 _(Chạy thực tế với Local Embedder `all-MiniLM-L6-v2` và Agent API `qwen3-coder-plus` - Vẫn khá hạn chế về mặt ngữ nghĩa tiếng Việt đối với các query ngắn)._

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> _Viết 2-3 câu:_
> Học được cách quản lý edge case khi chia cắt text bằng RegEx cho từng câu (`SentenceChunker`), giúp cho việc xử lý dữ liệu đa dạng và linh hoạt hơn nhiều thay vì chỉ đơn thuần là dùng split theo ký tự có sẵn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> _Viết 2-3 câu:_
> Thấy được mức độ ảnh hưởng quyết định của việc lựa chọn Embedding Model (vd: OpenAI vs Local SentenceTransformers vs Mock) đối với độ nhạy của Agent. Hóa ra Chunking chỉ góp phần định dạng đầu vào nhưng Embedding Model mới thực sự là "linh hồn" của thuật toán search ngữ nghĩa.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> _Viết 2-3 câu:_
> Tôi muốn cải tiến khâu extract data để bóc tách rõ tên "Chương" và từng "Danh mục" bằng RegExp rồi gán vào Metadata. Sau này ứng dụng tính năng `search_with_filter` của `EmbeddingStore` lọc Metadata trước khi tính toán similarity sẽ khoanh vùng tốt hơn và cải thiện chất lượng Agent.

---

## Tự Đánh Giá

| Tiêu chí               | Điểm  | Lý do ngắn                             |
| ---------------------- | ----- | -------------------------------------- |
| Warm-up                | 5/5   | đúng + giải thích rõ                   |
| Document selection     | 8/10  | thiếu số lượng doc                     |
| Chunking strategy      | 13/15 | phân tích tốt, thiếu phản biện Agentic |
| My approach            | 10/10 | rất rõ, đúng kỹ thuật                  |
| Similarity predictions | 4/5   | có phân tích nhưng chưa sâu            |
| Results                | 9/10  | có benchmark, thiếu sample answer      |
| Core implementation    | 30/30 | 43/43 test pass                        |
| Demo                   | 5/5   | đầy đủ insight                         |

Tôổn: 5 + 8 + 13 + 10 + 4 + 9 + 30 + 5 = 84 / 100
