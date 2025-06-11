import argparse
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a translation model for subtitles")
    parser.add_argument('--train_file', type=str, default='data/train.csv',
                        help='Path to training CSV file with columns source,target')
    parser.add_argument('--valid_file', type=str, default='data/val.csv',
                        help='Path to validation CSV file with columns source,target')
    parser.add_argument('--model_name', type=str, default='Helsinki-NLP/opus-mt-zh-en',
                        help='Pretrained model name or path')
    parser.add_argument('--output_dir', type=str, default='model', help='Where to store the trained model')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    data_files = {'train': args.train_file, 'validation': args.valid_file}
    dataset = load_dataset('csv', data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(example):
        inputs = tokenizer(example['source'], truncation=True)
        targets = tokenizer(example['target'], truncation=True)
        example['input_ids'] = inputs['input_ids']
        example['attention_mask'] = inputs['attention_mask']
        example['labels'] = targets['input_ids']
        return example

    tokenized = dataset.map(preprocess, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        logging_dir='logs'
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
