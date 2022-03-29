from argparse import ArgumentParser
from re import L
from torch.utils.checkpoint import checkpoint
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_steps", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_tokens("\n", special_tokens=True)
    tokenizer.save_pretrained(args.output)

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    train_dataset=LineByLineTextDataset(tokenizer=tokenizer, file_path=args.corpus, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.output, overwrite_output_dir=True, num_train_epochs=args.epoch, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, save_total_limit=5, warmup_steps=args.warmup_steps, weight_decay=0.01,
        adam_beta2=0.98, adam_epsilon=1e-6, save_strategy="epoch", fp16=True, gradient_checkpointing=True,
        )# save_steps=10000

    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)


    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
