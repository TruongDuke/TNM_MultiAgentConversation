# TNM Cancer Staging với Multi-Agent Collaboration

Dự án sử dụng Multi-Agent Collaboration (MAC) với AutoGen để phân giai đoạn TNM ung thư phổi từ mô tả lâm sàng và hình ảnh. Hệ thống sử dụng nhiều agents (bác sĩ chuyên khoa) thảo luận và đưa ra chẩn đoán TNM thông qua cơ chế voting và confidence scoring.

## 📋 Tổng quan

Dự án này triển khai một hệ thống AI để chẩn đoán giai đoạn TNM (Tumor, Node, Metastasis) cho ung thư phổi dựa trên:
- Mô tả lâm sàng và hình ảnh từ bệnh án
- Multi-agent collaboration với AutoGen
- Voting mechanism để tổng hợp ý kiến từ nhiều agents
- Confidence scoring để đánh giá độ tin cậy của kết quả

## ✨ Tính năng chính

### 1. Multi-Agent Collaboration
- **main_ws.py**: Workflow với Supervisor điều phối và Consultant chọn chuyên khoa
- **main_woexpert_tnm.py**: Workflow đơn giản không có Supervisor, các doctors tự thảo luận

### 2. Voting Mechanism
- Thu thập tất cả proposals từ mọi agent
- Weighted voting với confidence multiplier
- Phát hiện disagreement giữa các agents

### 3. Confidence Scoring
- Đánh giá confidence dựa trên độ rõ ràng của case description
- Tính confidence từ agreement giữa các agents
- Aggregate confidence cho kết quả cuối cùng

### 4. Local LLM Server
- Hỗ trợ chạy Llama-3.1-8B-Instruct local qua FastAPI
- Tương thích với OpenAI API format

## 📁 Cấu trúc dự án

```
TNM_MAC/
├── main_ws.py                 # Main script với Supervisor
├── main_woexpert_tnm.py       # Main script không có Supervisor
├── qwen_server.py             # Local Qwen-2.5 3B LLM server (FastAPI)
├── requirements.txt           # Dependencies
├── configs/
│   └── config_list.json       # Model configuration
├── dataset/
│   └── tnm_cases.json         # Dataset TNM cases (1408 cases)
├── utils/
│   ├── __init__.py
│   ├── data.py                # Dataset loader
│   ├── prompts.py             # System prompts cho agents
│   ├── utils.py               # Utility functions
│   ├── voting.py              # Voting mechanism
│   └── confidence.py         # Confidence scoring
└── output/                    # Kết quả output
    ├── MAC_WS/               # Output từ main_ws.py
    └── MAC_WOEXPERT_TNM/     # Output từ main_woexpert_tnm.py
```

## 🔧 Yêu cầu hệ thống

- Python 3.8+
- CUDA (nếu chạy local LLM)
- RAM: Tối thiểu 16GB (để chạy Llama-3.1-8B)
- Disk: ~20GB (cho model và dataset)

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd TNM_MAC
```

### 2. Tạo virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# hoặc
myenv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình HuggingFace token (nếu dùng local LLM)

```bash
export HF_TOKEN="your_huggingface_token_here"
```

## ⚙️ Cấu hình

### Model Configuration (`configs/config_list.json`)

```json
[
  {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "api_key": "NotRequired",
    "base_url": "http://127.0.0.1:4000",
    "tags": ["x_llama3"]
  }
]
```

### Dataset Format (`dataset/tnm_cases.json`)

```json
{
  "Cases": [
    {
      "Type": "TNM",
      "Final Name": "{T:T4,N:N3,M:M0}",
      "Case URL": "56344",
      "Initial Presentation": "左上葉全体が無気肺になっています...",
      "Follow-up Presentation": "",
      "Meta": {
        "split": "Train",
        "T": 4,
        "N": 3,
        "M": 0
      }
    }
  ]
}
```

## 🚀 Cách sử dụng

### 1. Khởi động Local LLM Server (nếu dùng local model)

```bash
python qwen_server.py
```

Server sẽ chạy tại `http://127.0.0.1:4000`

### 2. Chạy với Supervisor (`main_ws.py`)

```bash
python main_ws.py \
    --model_name x_llama3 \
    --dataset_name tnm_cases \
    --num_specialists 3 \
    --n_round 9 \
    --times 1
```

**Arguments:**
- `--model_name`: Model tag (default: `x_llama3`)
- `--dataset_name`: Dataset name (default: `tnm_cases`)
- `--num_specialists`: Số lượng specialists (default: 3)
- `--n_round`: Số rounds trong group chat (default: 9)
- `--times`: Số lần lặp lại experiment (default: 1)
- `--output_dir`: Thư mục output (default: `output`)

### 3. Chạy không có Supervisor (`main_woexpert_tnm.py`)

```bash
python main_woexpert_tnm.py \
    --model_name x_llama3 \
    --dataset_name tnm_cases \
    --num_doctors 3 \
    --n_round 10 \
    --times 1
```

**Arguments:**
- `--num_doctors`: Số lượng doctors (default: 3)
- Các arguments khác tương tự `main_ws.py`

## 🎯 Voting Mechanism & Confidence Scoring

### Workflow

1. **Extract Proposals**: Thu thập tất cả T, N, M từ mọi agent trong chat history
2. **Calculate Confidence**: 
   - Từ case description (size, invasion, lymph nodes, metastasis)
   - Từ agreement với các agents khác
3. **Weighted Voting**: 
   - Mỗi proposal có weight = base_weight × confidence_multiplier
   - Confidence multiplier: high=1.5, medium=1.0, low=0.5
   - Supervisor có base_weight=1.5, doctors=1.0
4. **Output**: Kết quả cuối cùng + consensus score + confidence + disagreements

### Confidence Levels

- **High**: Thông tin rõ ràng và chắc chắn
  - Ví dụ: Tumor size được đề cập rõ ràng, lymph node location cụ thể
- **Medium**: Thông tin có nhưng hơi mơ hồ
  - Ví dụ: Lymph nodes được đề cập nhưng location không rõ
- **Low**: Thông tin thiếu hoặc không rõ ràng
  - Ví dụ: Không có thông tin về tumor size

## 📊 Output Format

### File JSON kết quả (`{case_crl}.json`)

```json
{
  "Type": "TNM",
  "Crl": "56344",
  "Name": "{T:T4,N:N3,M:M0}",
  "Presentation": "左上葉全体が無気肺になっています...",
  "Cost": 0.0,
  "T": 4,
  "N": 3,
  "M": 0,
  "TNM": "T4,N3,M0",
  "Rationale": "Tumor size 74mm > 7cm (T4)...",
  "Areas of Disagreement": "None",
  "Consensus_Score": 0.92,
  "Factor_Consensus": {
    "T": 0.90,
    "N": 0.95,
    "M": 1.00
  },
  "Confidence": {
    "T": "high",
    "N": "high",
    "M": "high"
  },
  "Disagreements": ["None"],
  "Num_Proposals": 4
}
```

### File Conversation (`{case_crl}_conversation.json`)

Lưu toàn bộ hội thoại giữa các agents để phân tích.

## 📈 Output Directory Structure

```
output/
├── MAC_WS/                    # Từ main_ws.py
│   └── tnm/
│       └── x_llama3/
│           └── {num_specialists}-{n_round}/
│               └── {times}/
│                   ├── {case_crl}.json
│                   └── {case_crl}_conversation.json
│
└── MAC_WOEXPERT_TNM/          # Từ main_woexpert_tnm.py
    └── tnm/
        └── x_llama3/
            └── {num_doctors}-{n_round}/
                └── {times}/
                    ├── {case_crl}.json
                    └── {case_crl}_conversation.json
```

## 🔍 TNM Staging Rules

Hệ thống tuân theo các quy tắc TNM rút gọn cho ung thư phổi:

### T Factor
- **T1**: Size < 3 cm
- **T2**: Size 3–5 cm
- **T3**: Size 5–7 cm OR local invasion (chest wall, parietal pericardium, phrenic nerve)
- **T4**: Size > 7 cm OR invasion to mediastinum, trachea, heart/great vessels, esophagus, vertebra, carina

### N Factor
- **N0**: No regional lymph node metastasis
- **N1**: Ipsilateral peribronchial/hilar lymph nodes
- **N2**: Ipsilateral mediastinal/subcarinal lymph nodes
- **N3**: Contralateral mediastinal/hilar OR scalene/supraclavicular nodes

### M Factor
- **M0**: No distant metastasis
- **M1**: Distant metastasis

## 🧪 Ví dụ sử dụng

### Ví dụ 1: Chạy với 5 specialists và 10 rounds

```bash
python main_ws.py \
    --num_specialists 5 \
    --n_round 10 \
    --times 1
```

### Ví dụ 2: Chạy với custom output directory

```bash
python main_woexpert_tnm.py \
    --output_dir results \
    --num_doctors 4 \
    --n_round 12
```

### Ví dụ 3: Chạy nhiều lần để test reproducibility

```bash
python main_ws.py \
    --times 3 \
    --num_specialists 3
```

## 🐛 Troubleshooting

### Lỗi: "Không tìm thấy proposal TNM nào"

**Nguyên nhân**: Agents không output JSON format đúng

**Giải pháp**:
- Kiểm tra prompts trong `utils/prompts.py`
- Tăng `--n_round` để agents có thời gian thảo luận nhiều hơn
- Kiểm tra conversation file để xem agents đã nói gì

### Lỗi: Connection refused khi gọi LLM server

**Nguyên nhân**: LLM server chưa chạy hoặc sai port

**Giải pháp**:
- Đảm bảo `qwen_server.py` đang chạy
- Kiểm tra `base_url` trong `configs/config_list.json` đúng với server

### Lỗi: Out of memory khi load model

**Nguyên nhân**: Model quá lớn cho GPU/RAM

**Giải pháp**:
- Giảm batch size
- Sử dụng model nhỏ hơn
- Sử dụng quantization (8-bit, 4-bit)

## 📝 Notes

- Mặc định chỉ chạy 10 cases đầu để test (`min(10, data_len)`)
- Để chạy full dataset, sửa `min(10, data_len)` thành `data_len` trong code
- Temperature settings:
  - Consultant: 0 (deterministic)
  - Doctors/Supervisor: 1 (creative)
- Voting mechanism tự động skip các cases đã có kết quả

## 🔬 Phân tích kết quả

### Consensus Score
- **> 0.8**: Consensus tốt, kết quả đáng tin
- **0.6-0.8**: Consensus trung bình, có một số disagreement
- **< 0.6**: Consensus thấp, cần xem lại case

### Confidence
- **High**: Kết quả đáng tin, thông tin rõ ràng
- **Medium**: Kết quả có thể đúng nhưng cần xem xét
- **Low**: Kết quả không chắc chắn, case description thiếu thông tin

### Disagreements
- Nếu có disagreements, xem `All_Proposals` trong conversation để phân tích
- Các agents disagree thường do case description mơ hồ

## 📚 Tài liệu tham khảo

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [TNM Classification](https://www.cancer.gov/about-cancer/diagnosis-staging/staging)
- [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## 📄 License

[Thêm license nếu có]

## 👥 Contributors

[Thêm contributors nếu có]

## 🙏 Acknowledgments

- AutoGen team tại Microsoft
- HuggingFace cho model Llama-3.1-8B-Instruct

---

**Lưu ý**: Đây là dự án nghiên cứu, không nên sử dụng cho mục đích lâm sàng thực tế mà không có sự giám sát của chuyên gia y tế.

