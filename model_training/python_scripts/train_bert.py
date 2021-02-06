from transformers import BertForSequenceClassification, BertTokenizer, \
    Trainer, TrainingArguments, EarlyStoppingCallback
from MongoDataLoader import get_title_dataset

train_batch_size = 8
val_batch_size = 32

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pretrain_dataset, train_dataset, val_dataset = get_title_dataset('physics', tokenizer, partition_fractions=[0.3, 0.6, 0.1])

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

def train_model(model, train_dataset):
    train_args = TrainingArguments(output_dir='checkpoints',
                                   evaluation_strategy='steps',
                                   eval_steps=len(val_dataset)//train_batch_size,
                                   num_train_epochs=1,
                                   load_best_model_at_end=True,
                                   per_device_train_batch_size=train_batch_size,
                                   per_device_eval_batch_size=val_batch_size,
                                   disable_tqdm=True,
                                   save_steps=100000,
                                   logging_steps=100000,
                                   fp16=True,
                                   save_total_limit=4)
    #early_stopping = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-3)
    trainer = Trainer(model, train_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()

print('Training output layer only...')
freeze_model(model)
train_model(model, pretrain_dataset)
print('Training all layers...')
unfreeze_model(model)
train_model(model, train_dataset)

model.save_pretrained('BERT_pytorch_model')
