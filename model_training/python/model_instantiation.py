import torch
from transformers import BertForSequenceClassification, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Keyword arguments to pass to the tokenizer.
tokenizer_args = {'truncation': True,
                  'padding': True,
                  'return_tensors': 'pt'}


class ClassificationCollateFn:
    def __init__(self, inputs_col, labels_col, tokenizer):
        self.inputs_col = inputs_col
        self.labels_col = labels_col
        self.tokenizer = tokenizer

    def __call__(self, batch):
        raw_inputs = [row[self.inputs_col] for row in batch]
        labels = [row[self.labels_col] for row in batch]
        inputs = self.tokenizer(raw_inputs, **tokenizer_args)
        inputs['labels'] = torch.tensor(labels).float()
        return inputs


class SummarizationCollateFn:
    def __init__(self, inputs_col, outputs_col, tokenizer, prefix='summarize: '):
        self.inputs_col = inputs_col
        self.outputs_col = outputs_col
        self.tokenizer = tokenizer
        self.prefix = prefix

    def __call__(self, batch):
        raw_inputs = [self.prefix + row[self.inputs_col] for row in batch]
        raw_outputs = [row[self.outputs_col] for row in batch]
        inputs = self.tokenizer(raw_inputs, **tokenizer_args)
        outputs = self.tokenizer(raw_outputs, **tokenizer_args)
        output_ids = outputs['input_ids']
        # T5 expects labels to be padded with -100, not 0, so that it ignores them when computing the loss.
        output_ids[output_ids == 0] = -100
        inputs['labels'] = output_ids
        return inputs


def freeze_model(model):
    """Freeze all layers except the final classification layer."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_model(model):
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


def get_bert_model(name='bert-base-uncased', frozen=False):
    """Instantiates the model and collation function for BERT."""
    model = BertForSequenceClassification.from_pretrained(name, num_labels=1)
    tokenizer = BertTokenizer.from_pretrained(name)
    collate_fn = ClassificationCollateFn(inputs_col='Title', labels_col='Answered', tokenizer=tokenizer)
    if frozen:
        freeze_model(model)
    return model, collate_fn


def get_t5_model(name='t5-small'):
    """Instantiates the model and collation function for T5."""
    model = T5ForConditionalGeneration.from_pretrained(name, num_labels=1)
    tokenizer = T5Tokenizer.from_pretrained(name)
    collate_fn = SummarizationCollateFn(inputs_col='Body', outputs_col='Title', tokenizer=tokenizer)
    return model, collate_fn