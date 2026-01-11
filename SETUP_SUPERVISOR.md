# Hướng dẫn Setup Supervisor Fine-tuned Model

## Tổng quan

Model đã được fine-tune trên Google Colab và được tích hợp vào dự án để Supervisor agent sử dụng.

## Cấu trúc

```
TNM_MultiAgentConversation/
├── models/
│   └── llama-tnm-lora/          # Model fine-tune (đã download)
├── supervisor_server.py         # Server cho fine-tuned model
├── configs/
│   └── config_list.json         # Đã thêm supervisor_finetuned
└── main_ws.py                   # Đã cập nhật để dùng supervisor model
```

## Cài đặt

### 1. Install dependencies

```bash
pip install peft
```

### 2. Set HuggingFace Token

```bash
export HF_TOKEN="hf_your_token_here"
```

Hoặc tạo file `.env`:
```
HF_TOKEN=hf_your_token_here
```

## Sử dụng

### Bước 1: Khởi động Servers

**Terminal 1: Server cho doctors (base model)**
```bash
python qwen_server.py
# Hoặc llama31_8b_server.py
# Chạy trên port 4000
```

**Terminal 2: Server cho supervisor (fine-tuned model)**
```bash
export HF_TOKEN="hf_your_token_here"  # Nếu chưa set
python supervisor_server.py
# Chạy trên port 4001
```

### Bước 2: Chạy Main Script

```bash
python main_ws.py \
    --model_name x_llama3 \
    --supervisor_model_name supervisor_finetuned \
    --num_specialists 3 \
    --n_round 9 \
    --times 1
```

## Arguments

- `--model_name`: Model cho doctors (default: `x_llama3`)
- `--supervisor_model_name`: Model cho supervisor (default: `supervisor_finetuned`)
- Các arguments khác giữ nguyên

## Kiểm tra

### Test server supervisor

```bash
curl http://127.0.0.1:4001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Test"}],
    "max_tokens": 50
  }'
```

## Troubleshooting

### Lỗi: Model not found
- Kiểm tra `models/llama-tnm-lora/` có đầy đủ files
- Kiểm tra đường dẫn trong `supervisor_server.py`

### Lỗi: HF_TOKEN not found
- Set environment variable: `export HF_TOKEN="hf_..."`
- Hoặc thêm vào `.env` file

### Lỗi: Connection refused
- Đảm bảo `supervisor_server.py` đang chạy
- Kiểm tra port 4001 không bị chiếm

## Notes

- Supervisor dùng fine-tuned model (port 4001)
- Doctors dùng base model (port 4000)
- Model fine-tune được train với 108 TNM cases, 3 epochs
- Training loss giảm từ 2.56 → 1.55



