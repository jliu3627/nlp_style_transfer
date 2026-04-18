from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import pandas as pd
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_NAME = "t5-small"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 128

# Example: replace with your real ParaDetox file loading
df = pd.DataFrame({
    "source": [
        "you are an idiot",
        "this idea is stupid",
        "shut up and leave"
    ],
    "target": [
        "I disagree with what you said.",
        "I do not think this idea is a good one.",
        "Please stop talking and leave."
    ]
})

dataset = Dataset.from_pandas(df)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to(device)

def preprocess(example):
    model_input = "rewrite toxic to neutral: " + example["source"]
    inputs = tokenizer(
        model_input,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        text_target=example["target"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

args = Seq2SeqTrainingArguments(
    output_dir="./rewrite_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    predict_with_generate=True,
    fp16=False,
    dataloader_pin_memory=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    # tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Example inference
text = "rewrite toxic to neutral: you are useless"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
output_ids = model.generate(**inputs, max_length=64)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))