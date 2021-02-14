import pymongo
import pandas as pd
from datetime import datetime
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split






def get_mongo_dataset(collection, indices, projection, chunk_size=-1):
    if chunk_size == -1:
        return MongoDataset(collection, indices, projection)
    else:
        return MongoDataset(collection, indices, projection, chunk_size)





def get_dataset(forum, year, mode, val_size):
    posts = get_mongo_collection(forum)
    query = single_year_query(year)
    if mode == 'bert':

    else:
        raise Exception(f"Unrecognized mode: '{mode}'. Should be 'bert' or 't5'.")
    ids = get_mongo_ids(posts, query)
    train_ids, val_ids = train_test_split(ids, test_size=val_size)
    # Training data will be streamed, but validation data will stay in memory.
    train_dataset = MongoIterableDataset(posts, train_ids, projection)
    val_dataset = MongoDataset(posts, val_ids, projection)
    return train_dataset, val_dataset

