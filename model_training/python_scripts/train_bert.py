from transformers import BertForSequenceClassification, BertTokenizer, \
    Trainer, TrainingArguments, EarlyStoppingCallback
from data_loading import get_mongo_dataset, SequenceClassificationCollateFn
import os

forum = 'overflow'
year = 2018
os.chdir(forum)
if not os.path.exists('BERT'):
    os.mkdir('BERT')
os.chdir('BERT')

batch_size = 64
steps_between_val = 16
pretrain_steps = 256
train_steps = 1024
val_size = steps_between_val * batch_size * 8

# steps_between_val = 1_000
# pretrain_steps = 10_000
# train_steps = 10_000
# val_size = 10_000

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
collate_fn = SequenceClassificationCollateFn(inputs_col='Title', labels_col='Answered', tokenizer=tokenizer)

train_dataset, val_dataset = get_mongo_dataset(forum=forum, year=year, mode='bert', val_size=val_size)

# Load the model
print('Loading model...')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)


def freeze_model(model):
    # Freeze all layers except the last at first.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_model(model):
    # Unfreeze all parameters.
    for p in model.parameters():
        p.requires_grad = True


def train_model(model, train_dataset, steps):
    train_args = TrainingArguments(output_dir='checkpoints',
                                   evaluation_strategy='steps',
                                   eval_steps=steps_between_val,
                                   max_steps=steps,
                                   per_device_train_batch_size=batch_size,
                                   per_device_eval_batch_size=batch_size,
                                   disable_tqdm=True,
                                   save_steps=2 * steps,
                                   logging_steps=steps_between_val,
                                   fp16=True,
                                   save_total_limit=4)
    trainer = Trainer(model=model,
                      args=train_args,
                      data_collator=collate_fn,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset)
    trainer.train()


print('Training output layer only...')
freeze_model(model)
train_model(model, train_dataset, pretrain_steps)
print('Training all layers...')
unfreeze_model(model)
train_model(model, train_dataset, train_steps)

model.save_pretrained('BERT_pytorch_model')
