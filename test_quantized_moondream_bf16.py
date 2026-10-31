"""Opt-in legacy GPU smoke script; importing this module never loads a model."""


def main():
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from bitsandbytes.nn import Linear4bit

    MODEL_ID = "vikhyatk/moondream2"

    print("Loading Moondream2 with selective 4-bit quantization...")

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["vision", "region"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    quantized = [
        name for name, module in model.named_modules()
        if isinstance(module, Linear4bit)
    ]

    vision_quantized = [
        name for name in quantized
        if ".vision." in "." + name + "."
    ]

    print("Total 4-bit layers:", len(quantized))
    print("Quantized vision layers:", len(vision_quantized))

    if vision_quantized:
        print("WARNING: Some vision layers are still quantized.")
        print("Do not treat this as a successful test yet.")

    image_path = "test_frames/cam1_0.jpg"
    image = Image.open(image_path).convert("RGB")

    print("Testing image inference...")
    with torch.inference_mode():
        encoded = model.encode_image(image)
        answer = model.answer_question(
            encoded,
            "What is happening in this scene?",
            tokenizer,
        )

    print("\nSUCCESS: Image inference completed.")
    print("Model answer:", answer)

    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )


if __name__ == "__main__":
    main()
