from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Default arguments to use when initializing a TrainArguments object.
default_training_args = {'evaluation_strategy': 'no',
                         'disable_tqdm': True,
                         'save_steps': 0,
                         'logging_steps': 0,
                         'fp16': True,
                         'save_total_limit': 1}




def get_train_args(**kwargs):
    """Initialize train arguments with slightly different defaults and options."""
    if 'batch_size' in kwargs:
        batch_size = kwargs.pop('batch_size')
        kwargs['per_device_train_batch_size'] = batch_size
        kwargs['per_device_eval_batch_size'] = batch_size
    head = kwargs.pop('head')
    if head == 'classification':
        trainer_args_class = TrainingArguments
    elif head == 'summarization':
        trainer_args_class = Seq2SeqTrainingArguments
    else:
        print(f"Unrecognized head type: {head}. Should be one of 'classification' or 'summarization'.")
        return
    train_args = trainer_args_class(**{**default_training_args, **kwargs})
    return train_args


def train_evaluate_save(trainer, return_predictions=False, save_model=True):
    trainer.train()
    if return_predictions:
        results = trainer.predict(trainer.eval_dataset)
    else:
        results = trainer.evaluate()
    if save_model:
        trainer.save_model()
    return results
