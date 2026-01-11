import os

# Tạo thư mục models
models_dir = "models/llama-tnm-lora"
os.makedirs(models_dir, exist_ok=True)

print(f"✅ Created directory: {models_dir}")
print(f"\n📁 Next steps:")
print(f"   1. Download model from Google Drive (llama-tnm-lora folder)")
print(f"   2. Extract all files into: {models_dir}/")
print(f"   3. Files should include:")
print(f"      - adapter_config.json")
print(f"      - adapter_model.bin")
print(f"      - tokenizer files")
print(f"\n📝 See {models_dir}/README.md for more details")



