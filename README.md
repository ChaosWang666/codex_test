# Subtitle Translation Training

This repository contains a simple example of how to fine-tune a machine translation model for TV subtitles using [Hugging Face Transformers](https://huggingface.co/transformers/).

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset in CSV format with the columns `source` and `target`. A tiny example is provided in the `data/` directory.

3. Run training:
   ```bash
   python scripts/train.py \
       --train_file data/train.csv \
       --valid_file data/val.csv \
       --model_name Helsinki-NLP/opus-mt-zh-en \
       --output_dir model
   ```

The script will fine-tune the specified model and save it to the `model/` directory.
