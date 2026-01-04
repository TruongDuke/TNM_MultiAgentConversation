import os
from typing import List, Optional
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH = "models/llama-tnm-lora"  # Đường dẫn đến LoRA adapters

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),  # Cần HF_TOKEN
    trust_remote_code=True,
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# Merge LoRA để inference nhanh hơn (optional - có thể bỏ qua nếu muốn giữ riêng)
# model = model.merge_and_unload()

model.eval()
print("Model loaded successfully!")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

    class Config:
        extra = "ignore"  # bỏ qua field thừa

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    input_ids = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": req.max_tokens or 512,
    }

    # Nếu temperature <= 0 => chuyển sang greedy decoding (do_sample=False)
    if req.temperature is None or req.temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = req.temperature
        generation_kwargs["top_p"] = req.top_p

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            **generation_kwargs,
        )

    generated = output_ids[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "id": "chatcmpl-llama-tnm-lora",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4001)  # Port khác với qwen_server


