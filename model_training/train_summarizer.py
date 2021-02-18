from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration, T5Tokenizer
import mongo_dataset

# Projection for loading the dataset.
projection = {'Title': True,
              'Body': True,
              '_id': False}

# Keyword arguments to pass to the tokenizer.
tokenizer_args = {'truncation': True,
                  'padding': True,
                  'return_tensors': 'pt'}

# Keyword arguments to pass to the trainer.
default_training_args = {'output_dir': 'T5',
                         'evaluation_strategy': 'no',
                         'disable_tqdm': False,
                         'save_steps': 0,
                         'logging_steps': 0,
                         'fp16': True,
                         'save_total_limit': 1,
                         'per_device_train_batch_size': 16,
                         'per_device_eval_batch_size': 16,
                         'num_train_epochs': 1,
                         }


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


def get_t5_model(name='t5-small'):
    """Instantiates the model and collation function for T5."""
    model = T5ForConditionalGeneration.from_pretrained(name, num_labels=1)
    tokenizer = T5Tokenizer.from_pretrained(name)
    collate_fn = SummarizationCollateFn(inputs_col='Body', outputs_col='Title', tokenizer=tokenizer)
    return model, collate_fn


def get_dataset(verbose=True, max_size=None):
    if verbose:
        print('Connecting to database...')
    dataset = mongo_dataset.MongoDataset()
    if verbose:
        print('Loading datasets...')
    train_dataset = dataset.get_partition('summarization_train', projection)
    val_dataset = dataset.get_partition('summarization_val', projection)
    if max_size:
        train_dataset = train_dataset[:max_size]
        val_dataset = val_dataset[:max_size]
    return train_dataset, val_dataset


def train_evaluate(model, collate_fn, train_dataset, val_dataset, **kwargs):
    train_args = Seq2SeqTrainingArguments(**{**default_training_args, **kwargs})
    trainer = Seq2SeqTrainer(model=model,
                             args=train_args,
                             data_collator=collate_fn,
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset)
    trainer.train()
    results = trainer.evaluate()
    return results


def main():
    train_dataset, val_dataset = get_dataset()
    model, collate_fn = get_t5_model()
    results = train_evaluate(model, collate_fn, train_dataset, val_dataset)
    print(results)
    model.save_pretrained(default_training_args['output_dir'])


if __name__ == '__main__':
    main()
