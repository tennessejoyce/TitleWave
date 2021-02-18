from partition_dataset import get_partition, get_mongo_collection
from transformers import Trainer, TrainingArguments


default_training_args = {'output_dir': 'BERT',
                         'evaluation_strategy': 'no',
                         'disable_tqdm': False,
                         'save_steps': 0,
                         'logging_steps': 0,
                         'fp16': True,
                         'save_total_limit': 1,
                         'per_device_train_batch_size': 32,
                         'per_device_eval_batch_size': 32,
                         'num_train_epochs': 1
                         }



def get_datasets():
    forum = 'overflow'
    collection = get_mongo_collection(forum)
    classifier_train = get_partition(collection, 'classification_train')

def get_trainer():


def main():

    train_args = TrainingArguments(default_training_args)
    model_init, collate_fn = get_bert_model(return_init=True, frozen=True)
    trainer = Trainer(model_init=model_init,
                      args=get_train_args(**base_hyperparameters),
                      data_collator=collate_fn,
                      train_dataset=classifier_train,
                      eval_dataset=classifier_val)


if __name__=='__main__':
    main()