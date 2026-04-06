"""Export the Charsiu G2P T5 model to ONNX format.

This script converts the PyTorch T5ForConditionalGeneration model to ONNX
using HuggingFace Optimum. It produces three ONNX graphs:
  - encoder_model.onnx
  - decoder_model.onnx
  - decoder_with_past_model.onnx (KV-cache for faster autoregressive decoding)

Requirements:
    pip install torch transformers optimum[onnxruntime]

Usage:
    python scripts/export_onnx.py [--output-dir onnx_export/] [--push-to-hub REPO_ID]

References:
    Original model: https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100
    CharsiuG2P: https://github.com/lingjzhu/CharsiuG2P

    @article{zhu2022charsiu-g2p,
      title={ByT5 model for massively multilingual grapheme-to-phoneme conversion},
      author={Zhu, Jian and Zhang, Cong and Jurgens, David},
      url={https://arxiv.org/abs/2204.03067},
      doi={10.48550/ARXIV.2204.03067},
      year={2022}
    }
"""

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

PYTORCH_MODEL_ID = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
TOKENIZER_MODEL_ID = "google/byt5-small"


def main():
    parser = argparse.ArgumentParser(description="Export Charsiu G2P model to ONNX")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_export",
        help="Directory to save ONNX files (default: onnx_export/)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID to push to (e.g. klebster/g2p_multilingual_byT5_tiny_onnx)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {PYTORCH_MODEL_ID} to ONNX...")
    model = ORTModelForSeq2SeqLM.from_pretrained(PYTORCH_MODEL_ID, export=True)
    model.save_pretrained(output_dir)
    print(f"ONNX files saved to {output_dir}/")

    # Validate the exported model
    print("Validating exported model...")
    model = ORTModelForSeq2SeqLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)

    inputs = tokenizer(
        "<eng-us>: hello", padding=True, add_special_tokens=False, return_tensors="pt"
    )
    preds = model.generate(**inputs, num_beams=1, max_length=50)
    result = tokenizer.decode(preds[0], skip_special_tokens=True)
    print(f"  Validation: '<eng-us>: hello' -> '{result}'")
    assert result == "ˈhɛɫoʊ", f"Expected 'ˈhɛɫoʊ', got '{result}'"
    print("  Validation passed!")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_folder(folder_path=str(output_dir), repo_id=args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
