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


class MongoDataset(torch.utils.data.Dataset):
    """Wraps a MongoDB collection as a Pytorch Dataset."""

    def __init__(self, collection: pymongo.collection.Collection, indices, projection):
        self.results = list(collection.find({'_id': {'$in': indices}}, projection))

    def __getitem__(self, idx):
        """Retrieve a single document from the MongoDB collection."""
        return self.results[idx]

    def __len__(self):
        return len(self.results)


class MongoIterableDataset(torch.utils.data.IterableDataset):
    """
    Wraps a MongoDB collection as a Pytorch IterableDataset, which produces documents at random from a specified
    subset of the collection. This offers a considerable speedup over MongoDataset because the documents can be
    requested in chunks, rather than one at a time.
    """

    def __init__(self, collection: pymongo.collection.Collection, indices, projection,
                 chuck_size: int = 256, shuffle: bool = True):
        self.collection = collection
        self.indices = indices
        self.projection = projection
        # Integers indexing the list of indices.
        meta_indices = np.arange(0, len(self.indices))
        if shuffle:
            np.random.shuffle(meta_indices)
        num_chunks = len(self.indices) // chuck_size
        chunked_meta_indices = np.array_split(meta_indices, num_chunks)
        self.chunked_indices = [[indices[i] for i in chunk] for chunk in chunked_meta_indices]

    def __iter__(self):
        """Return a generator that requests documents in chunks."""
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


def get_mongo_collection(forum):
    """Returns the Mongo collection corresponding to the specified StackExchange forum."""
    client = pymongo.MongoClient()
    posts = client.titlewave[f'{forum}.posts']
    return posts


def get_mongo_indices(collection, query):
    result = collection.find(query, {'_id': True})
    indices = [row['_id'] for row in result]
    return indices


def get_title_dataset(forum, val_fraction):
    posts = get_mongo_collection(forum)
    query = mongo_query(start_date=datetime(2019, 1, 1),
                        end_date=datetime(2020, 1, 1),
                        exclude_closed=True)
    titles_projection = {'Title': True,
                         'Answered': {'$gt': ['$AnswerCount', 0]},
                         '_id': False}
    indices = get_mongo_indices(posts, query)
    train_indices, val_indices = train_test_split(indices, test_size=val_fraction)
    train_dataset = MongoIterableDataset(posts, train_indices, titles_projection)
    # Validation dataset should be fully loaded into memory.
    val_dataset = MongoDataset(posts, val_indices, titles_projection)
    return train_dataset, val_dataset


def get_t5_dataset(forum, val_fraction):
    posts = get_mongo_collection(forum)
    query = mongo_query(start_date=datetime(2019, 1, 1),
                        end_date=datetime(2020, 1, 1),
                        exclude_closed=True)
    projection = {'Title': True,
                  'Body': True,
                  '_id': False}
    indices = get_mongo_indices(posts, query)
    train_indices, val_indices = train_test_split(indices, test_size=val_fraction)
    train_dataset = MongoIterableDataset(posts, train_indices, projection)
    # Validation dataset should be fully loaded into memory.
    val_dataset = MongoDataset(posts, val_indices, projection)
    return train_dataset, val_dataset
