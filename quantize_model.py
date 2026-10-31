from awq import AutoAWQForCausalLM
from transformers import AutoProcessor

model_path = "Qwen/Qwen2-VL-2B-Instruct"
quant_path = "qwen2-vl-2b-awq"

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

print("Loading model...")
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="cuda")
processor = AutoProcessor.from_pretrained(model_path)

print("Quantizing (this may take a while on 4GB VRAM)...")
model.quantize(processor.tokenizer, quant_config=quant_config)

model.save_quantized(quant_path)
processor.save_pretrained(quant_path)
print(f"Saved quantized model to {quant_path}")




