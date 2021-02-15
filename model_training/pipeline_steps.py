from model_instantiation import *
from model_training import *
from dataset_partitioning import *
import optuna

classifier_hyperparameters = {'output_dir': 'BERT',
                              'num_train_epochs': 1,
                              'disable_tqdm': False,
                              'batch_size': 32,
                              'head': 'classification'}
summarizer_hyperparameters = {'output_dir': 'T5',
                              'num_train_epochs': 1,
                              'disable_tqdm': False,
                              'batch_size': 16,
                              'head': 'summarization'}


def train_classifier(collection, hyperparameters=classifier_hyperparameters):
    classifier_train = get_classifier_dataset_partition(collection, 'classification_train')
    classifier_val = get_classifier_dataset_partition(collection, 'classification_val')
    model, collate_fn = get_bert_model()
    trainer = Trainer(model=model,
                      args=get_train_args(**hyperparameters),
                      data_collator=collate_fn,
                      train_dataset=classifier_train,
                      eval_dataset=classifier_val)
    result = train_evaluate_save(trainer)
    return result


def train_summarizer(collection, hyperparameters=summarizer_hyperparameters):
    summarizer_train = get_summarizer_dataset_partition(collection, 'summarization_train')
    summarizer_val = get_summarizer_dataset_partition(collection, 'summarization_val')
    model, collate_fn = get_t5_model()
    trainer = Seq2SeqTrainer(model=model,
                             args=get_train_args(**hyperparameters),
                             data_collator=collate_fn,
                             train_dataset=summarizer_train,
                             eval_dataset=summarizer_val)
    return trainer
    result = train_evaluate_save(trainer)
    return result


def eval_loss_objective(results_dict):
    return results_dict['eval_loss']


def hyperparameter_optimization(collection, search_space, n_trials=8, base_hyperparameters=classifier_hyperparameters):
    classifier_train = get_classifier_dataset_partition(collection, 'classification_train')
    classifier_val = get_classifier_dataset_partition(collection, 'classification_val')
    model_init, collate_fn = get_bert_model(return_init=True, frozen=False)
    trainer = Trainer(model_init=model_init,
                      args=get_train_args(**base_hyperparameters),
                      data_collator=collate_fn,
                      train_dataset=classifier_train,
                      eval_dataset=classifier_val)
    result = trainer.hyperparameter_search(search_space, n_trials=n_trials, compute_objective=eval_loss_objective)
    return result


def hp_search_space(trial: optuna.trial):
    params = {}
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    params['per_device_train_batch_size'] = batch_size
    params['per_device_eval_batch_size'] = 128
    return params


def train_generator():
    pass
