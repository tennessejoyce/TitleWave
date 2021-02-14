from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Default arguments to use when initializing a TrainArguments object.
default_training_args = {'evaluation_strategy': 'no',
                         'disable_tqdm': True,
                         'save_steps': 0,
                         'logging_steps': 0,
                         'fp16': True,
                         'save_total_limit': 1}


def train_evaluate_save(model, collate_fn, train_dataset, val_dataset, head, return_predictions=False, save_model=True,
                        **kwargs):
    """Train a model, evaluate it on a validation set, save the model, and return the evaluation results."""
    # Alias the 'batch_size' argument so that we can use the shorter name.
    if 'batch_size' in kwargs:
        batch_size = kwargs.pop('batch_size')
        kwargs['per_device_train_batch_size'] = batch_size
        kwargs['per_device_eval_batch_size'] = batch_size
    if head == 'classification':
        trainer_class = Trainer
        trainer_args_class = TrainingArguments
    elif head == 'summarization':
        trainer_class = Seq2SeqTrainer
        trainer_args_class = Seq2SeqTrainingArguments
    else:
        print(f"Unrecognized head type: {head}. Should be one of 'classification' or 'summarization'.")
        return
    # Initialize the training arguments object with the keyword arguments provided, and the defaults defined above.
    train_args = trainer_args_class(**{**default_training_args, **kwargs})
    trainer = trainer_class(model=model,
                            args=train_args,
                            data_collator=collate_fn,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset)
    trainer.train()
    if return_predictions:
        results = trainer.predict(val_dataset)
    else:
        results = trainer.evaluate()
    if save_model:
        trainer.save_model()
    return results
