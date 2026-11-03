from PIL import Image


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "vikhyatk/moondream2"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Replace this with an actual image filename from Step 2
    image_path =  "/home/deepika/vlm-edge-github/data/loco/Images/subset-1/362356,7729.jpg"

    image = Image.open(image_path).convert("RGB")
    enc_image = model.encode_image(image)

    question = "How many pallets are visible in this image?"

    answer = model.answer_question(
        enc_image,
        question,
        tokenizer=tokenizer
    )

    print("\nQuestion:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
