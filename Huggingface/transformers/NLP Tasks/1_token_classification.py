from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch


raw_datasets = load_dataset("conll2003")
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names
model_checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


raw_datasets = raw_datasets.remove_columns(['pos_tags', 'chunk_tags', 'id'])
raw_datasets = raw_datasets.rename_column('ner_tags', 'labels')
raw_datasets = raw_datasets.rename_column('tokens', 'words')


def align_label_with_tokens(labels, word_ids):
    new_labels = []
    current_word_id = None
    for word_id in word_ids:
        if word_id != current_word_id:
            current_word_id = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            label = -100
            new_labels.append(label)
        else:
            label = labels[current_word_id]
            if label%2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels



def tokenize_and_align(batch):
    tokenized_inputs = tokenizer(batch['words'], is_split_into_words=True, truncation=True)
    all_labels = batch['labels']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_label_with_tokens(labels, word_ids))
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

dataset = raw_datasets.map(tokenize_and_align, batched=True, remove_columns=['words'])

import evaluate


metric = evaluate.load("seqeval")


def compute_metric(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,id2label=id2label, label2id=label2id)


from transformers import TrainingArguments


args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
)


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)



from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metric,
    tokenizer=tokenizer
)

# trainer.train()


from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW

dataset['train'] = dataset['train'].select(range(1000))
dataset['validation'] = dataset['validation'].select(range(100))

train_dataloader = DataLoader(dataset['train'],
                              shuffle=True,
                              collate_fn=data_collator,
                              batch_size=32)

eval_dataloader = DataLoader(
    dataset["validation"], collate_fn=data_collator, batch_size=32
)


optimizer = AdamW(model.parameters(), lr=args.learning_rate)


accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model,
                                                                          optimizer,
                                                                          train_dataloader,
                                                                          eval_dataloader)


from transformers import get_scheduler


num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.inference_mode():
            outputs = model(**batch)
        
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        
        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )


accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("bert-finetuned-ner-accelerate", save_function=accelerator.save)
