import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=os.environ.get("HF_TOKEN")
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=os.environ.get("HF_TOKEN")
)

model.eval()
print("Model loaded.")

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

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens or 512,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    generated = output_ids[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "id": "chatcmpl-local-llama31-8B",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)