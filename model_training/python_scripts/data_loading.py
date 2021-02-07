import pymongo
import pandas as pd
from datetime import datetime
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

tokenizer_args = {'truncation': True,
                  'padding': True,
                  'return_tensors': 'pt'}


def split_list(indices, chunk_size):
    """Splits a list into chunks of specified size."""
    # Integers indexing the list of indices.
    meta_indices = np.arange(0, len(indices))
    np.random.shuffle(meta_indices)
    num_chunks = len(indices) // chunk_size
    chunked_meta_indices = np.array_split(meta_indices, num_chunks)
    return [[indices[i] for i in chunk] for chunk in chunked_meta_indices]


class MongoDataset(torch.utils.data.Dataset):
    """Wraps a MongoDB collection as a Pytorch Dataset."""

    def __init__(self, collection: pymongo.collection.Collection, indices, projection):
        """"""
        self.results = list(collection.find({'_id': {'$in': indices}}, projection))

    def __getitem__(self, idx):
        """Retrieve a single document from the MongoDB collection."""
        return self.results[idx]

    def __len__(self):
        return len(self.results)


class MongoIterableDataset(torch.utils.data.IterableDataset):
    """Wraps a MongoDB collection as a Pytorch IterableDataset which retrieves documents at random."""

    def __init__(self, collection, indices, projection, chunk_size=256, shuffle=True):
        """"""
        self.collection = collection
        self.indices = indices
        self.projection = projection
        self.chunked_indices = split_list(indices, chunk_size=chunk_size)

    def __iter__(self):
        """Return a generator that requests documents in chunks, but returns them one at a time."""
        for chunk in self.chunked_indices:
            cursor = self.collection.find({'_id': {'$in': chunk}}, self.projection)
            rows = list(cursor)
            np.random.shuffle(rows)
            for row in rows:
                yield row

    def __len__(self):
        return len(self.indices)


class SequenceClassificationCollateFn:
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


class Seq2SeqCollateFn:
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


def mongo_query(start_date, end_date, exclude_closed):
    """Create a MongoDB query based on a set of conditions."""
    query = {}
    if start_date:
        if not ('CreationDate' in query):
            query['CreationDate'] = {}
        query['CreationDate']['$gte'] = start_date
    if end_date:
        if not ('CreationDate' in query):
            query['CreationDate'] = {}
        query['CreationDate']['$lt'] = end_date
    if exclude_closed:
        query['Closed'] = False
    return query

def single_year_query(year, exclude_closed=True):
    """Returns a MongoDB query returnins all posts for a given year."""
    query = mongo_query(start_date=datetime(2019, 1, 1),
                        end_date=datetime(2020, 1, 1),
                        exclude_closed=exclude_closed)
    return query


def get_mongo_collection(forum):
    """Returns the Mongo collection corresponding to the specified StackExchange forum."""
    client = pymongo.MongoClient()
    posts = client.titlewave[f'{forum}.posts']
    return posts


def get_mongo_ids(collection, query):
    result = collection.find(query, {'_id': True})
    ids = [row['_id'] for row in result]
    return ids


def get_mongo_dataset(forum, year, mode, val_size):
    posts = get_mongo_collection(forum)
    query = single_year_query(year)
    if mode == 'bert':
        projection = {'Title': True,
                     'Answered': {'$gt': ['$AnswerCount', 0]},
                     '_id': False}
    elif mode == 't5':
        projection = {'Title': True,
                      'Body': True,
                      '_id': False}
    else:
        raise Exception(f"Unrecognized mode: '{mode}'. Should be 'bert' or 't5'.")
    ids = get_mongo_ids(posts, query)
    train_ids, val_ids = train_test_split(ids, test_size=val_size)
    # Training data will be streamed, but validation data will stay in memory.
    train_dataset = MongoIterableDataset(posts, train_ids, projection)
    val_dataset = MongoDataset(posts, val_ids, projection)
    return train_dataset, val_dataset

