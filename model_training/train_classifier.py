from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer
import mongo_dataset
import torch
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

# Projection for loading the dataset.
projection = {'Title': True,
              'Answered': {'$gt': ['$AnswerCount', 0]},
              '_id': False}

# Keyword arguments to pass to the tokenizer.
tokenizer_args = {'truncation': True,
                  'padding': True,
                  'return_tensors': 'pt'}

# Keyword arguments to pass to the trainer.
default_training_args = {'output_dir': 'BERT',
                         'evaluation_strategy': 'no',
                         'disable_tqdm': False,
                         'save_steps': 0,
                         'logging_steps': 0,
                         'fp16': True,
                         'save_total_limit': 1,
                         'per_device_train_batch_size': 32,
                         'per_device_eval_batch_size': 32,
                         'num_train_epochs': 1,
                         }


class ClassificationCollateFn:
    def __init__(self, inputs_col, labels_col, tokenizer):
        self.inputs_col = inputs_col
        self.labels_col = labels_col
        self.tokenizer = tokenizer

    def __call__(self, batch):
        raw_inputs = [row[self.inputs_col] for row in batch]
        labels = [row[self.labels_col] for row in batch]
        inputs = self.tokenizer(raw_inputs, **tokenizer_args)
        inputs['labels'] = torch.tensor(labels).long()
        return inputs


def get_bert_model(name='bert-base-uncased'):
    """Instantiates the model and collation function for BERT."""
    tokenizer = BertTokenizer.from_pretrained(name)
    collate_fn = ClassificationCollateFn(inputs_col='Title', labels_col='Answered', tokenizer=tokenizer)
    model = BertForSequenceClassification.from_pretrained(name, num_labels=2)
    return model, collate_fn


def compute_metrics(eval_prediction):
    """Compute the ROC-AUC score for binary classification."""
    probabilities = softmax(eval_prediction.predictions, axis=1)[:, 1]
    labels = eval_prediction.label_ids
    metrics = {}
    metrics['roc_auc_score'] = roc_auc_score(labels, probabilities)
    return metrics


def get_dataset(verbose=True, max_size=None):
    if verbose:
        print('Connecting to database...')
    dataset = mongo_dataset.MongoDataset()
    if verbose:
        print('Loading datasets...')
    train_dataset = dataset.get_partition('classification_train', projection)
    val_dataset = dataset.get_partition('classification_val', projection)
    if max_size:
        train_dataset = train_dataset[:max_size]
        val_dataset = val_dataset[:max_size]
    return train_dataset, val_dataset


def freeze_model(model):
    """Freeze all layers except the final classification layer."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.bert.pooler.parameters():
        p.requires_grad = True


def unfreeze_model(model):
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


def train_evaluate(model, collate_fn, train_dataset, val_dataset, **kwargs):
    train_args = TrainingArguments(**{**default_training_args, **kwargs})
    trainer = Trainer(model=model,
                      args=train_args,
                      data_collator=collate_fn,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      compute_metrics=compute_metrics)
    trainer.train()
    results = trainer.evaluate()
    return results


def two_phase_train(model, collate_fn, train_dataset, val_dataset, lr1, lr2):
    """
    Train for one epoch only the last two layers, then for one epoch every layer.

    Parameters:
        trainer - The transformers.Trainer object containing the model to be trained
        lr1 - Learning rate for the first epoch
        lr2 - Learning rate for the second epoch
    """
    # Freeze all but the last two layers, and train for one epoch.
    freeze_model(model)
    print('Training phase 1...')
    results1 = train_evaluate(model, collate_fn, train_dataset, val_dataset, learning_rate=lr1)
    print(results1)
    # Unfreeze the model, and train for another epoch.
    unfreeze_model(model)
    print('Training phase 2...')
    results2 = train_evaluate(model, collate_fn, train_dataset, val_dataset, learning_rate=lr2)
    print(results2)
    return results1, results2


def main():
    train_dataset, val_dataset = get_dataset()
    model, collate_fn = get_bert_model()
    lr1 = 3e-4
    lr2 = 3e-5
    two_phase_train(model, collate_fn, train_dataset, val_dataset, lr1, lr2)
    model.save_pretrained(default_training_args['output_dir'])


if __name__ == '__main__':
    main()
