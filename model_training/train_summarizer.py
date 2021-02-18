from partition_dataset import get_partition

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



def