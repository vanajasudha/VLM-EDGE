"""Opt-in legacy GPU smoke script; importing this module never loads a model."""


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "vikhyatk/moondream2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Moondream2 loaded successfully on:", next(model.parameters()).device)


if __name__ == "__main__":
    main()
