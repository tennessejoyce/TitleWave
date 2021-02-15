from dataset_partitioning import *
from pipeline_steps import *
import os

redo_partitions = False
redo_classifier = True
redo_summarizer = False
redo_generator = False

forum = 'overflow'
posts = get_mongo_collection(forum)
os.chdir(forum)




if redo_partitions:
    make_partitions(posts, year_range_query(2017, 2019))
if redo_classifier:
    hyperparameter_optimization(posts, hp_search_space, n_trials=8)
if redo_summarizer:
    train_summarizer(posts)
if redo_generator:
    train_generator()
