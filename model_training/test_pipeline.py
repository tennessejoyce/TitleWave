from dataset_partitioning import *
from model_instantiation import *
from model_training import *
import os

redo_partitions = False
redo_classifier = False
redo_summarizer = True
redo_generator = True

forum = 'overflow'
posts = get_mongo_collection(forum)
os.chdir(forum)

classifier_hyperparameters = {'output_dir': 'BERT', 'num_train_epochs': 1, 'disable_tqdm': False, 'batch_size': 32}
summarizer_hyperparameters = {'output_dir': 'T5', 'num_train_epochs': 1, 'disable_tqdm': False, 'batch_size': 16}


def train_classifier(collection):
    classifier_train = get_classifier_dataset_partition(collection, 'classification_train')
    classifier_val = get_classifier_dataset_partition(collection, 'classification_val')
    model, collate_fn = get_bert_model()
    result = train_evaluate_save(model, collate_fn, classifier_train, classifier_val, head='classification',
                                 **classifier_hyperparameters)
    print(result)


def train_summarizer(collection):
    summarizer_train = get_summarizer_dataset_partition(collection, 'summarization_train')
    summarizer_val = get_summarizer_dataset_partition(collection, 'summarization_val')
    model, collate_fn = get_t5_model()
    result = train_evaluate_save(model, collate_fn, summarizer_train, summarizer_val, head='summarization',
                                 **summarizer_hyperparameters)
    print(result)


def train_generator():
    pass


if redo_partitions:
    make_partitions(posts, year_range_query(2017, 2019))
if redo_classifier:
    train_classifier(posts)
if redo_summarizer:
    train_summarizer(posts)
if redo_generator:
    train_generator()
