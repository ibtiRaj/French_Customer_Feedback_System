import torch
import numpy as np
from datasets import Dataset
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments
)

class CamembertSentimentModel:
    def __init__(self, model_name="camembert-base"):
        self.model_name = model_name
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model = CamembertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

    # -------------------------
    # TOKENIZATION
    # -------------------------
    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length"
        )

    def _prepare_dataset(self, texts, labels):
        dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })
        dataset = dataset.map(self._tokenize, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    # -------------------------
    # TRAINING
    # -------------------------
    def train(self, train_texts, train_labels, val_texts, val_labels):
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        val_dataset = self._prepare_dataset(val_texts, val_labels)

        training_args = TrainingArguments(
            output_dir="./camembert_results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_steps=50
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            acc = (preds == labels).mean()
            return {"accuracy": acc}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

    # -------------------------
    # PREDICTION
    # -------------------------
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)
        return preds.cpu().numpy().tolist()

    # -------------------------
    # SAVE / LOAD
    # -------------------------
    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.model = CamembertForSequenceClassification.from_pretrained(path)
        self.tokenizer = CamembertTokenizer.from_pretrained(path)