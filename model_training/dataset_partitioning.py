import numpy as np
import pymongo
from datetime import datetime, timedelta
from mongo_dataset import *
from tqdm import tqdm

# Split into three main sections: BERT, T5, Combined generation


default_proportions = {
    'classification_train': 0.05,
    'classification_val': 0.05,
    'classification_test': 0.05,
    'summarization_train': 0.25,
    'summarization_val': 0.25,
    'summarization_test': 0.25,
    'generation_train': 0.05,
    'generation_test': 0.05,
}


def get_mongo_ids(collection, query):
    result = collection.find(query, {'_id': True})
    ids = [row['_id'] for row in result]
    return ids


def split_to_proportions(array, proportions, shuffle=True):
    """Splits an array into several subarrays, which are given proportions of the whole."""
    assert np.sum(proportions) == 1
    # Sizes of the subarrays, rounded down.
    sizes = [int(len(array) * p) for p in proportions]
    # Divide the remainder among the subarrays from first to last.
    remainder = len(array) - np.sum(sizes)
    for i in range(remainder):
        sizes[i] += 1
    if shuffle:
        np.random.shuffle(array)
    split_locations = np.cumsum(sizes)[:-1]
    np_splits = np.split(array, split_locations)
    # Convert from a nparray back to a nested list of ints.
    return np_to_int_list(np_splits)


def np_to_int_list(ids_np):
    return [[int(i) for i in s] for s in ids_np]


def batch_update(collection, ids, command, batch_size=256):
    num_batches = len(ids) // batch_size
    splits = np_to_int_list(np.array_split(ids, num_batches))
    for batch_ids in tqdm(splits):
        collection.update_many({'_id': {'$in': batch_ids}}, command)


def make_partitions(collection, query, proportions=None):
    print('Reseting partitions...')
    reset_partitions(collection)
    print('Assigning ids to partitions...')
    if proportions is None:
        proportions = default_proportions
    filtered_ids = get_mongo_ids(collection, query)
    split_ids = split_to_proportions(filtered_ids, list(proportions.values()))
    print('Setting partitions in database...')
    for name, ids in zip(proportions.keys(), split_ids):
        print(f'{name}: {len(ids)} posts')
        batch_update(collection, ids, {'$set': {'partition': name}})
    # Create a sparse index on the partition field.
    print('Indexing partitions...')
    collection.create_index('partition', sparse=True)


def reset_partitions(collection):
    """Remove the partition field from all documents in the collection."""
    collection.update_many({'partition': {'$exists': True}}, {'$unset': {'partition': 1}})


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


def year_range_query(start_year, end_year, exclude_closed=True):
    """Returns a MongoDB query returnins all posts for a given year."""
    query = mongo_query(start_date=datetime(start_year, 1, 1),
                        end_date=datetime(end_year + 1, 1, 1),
                        exclude_closed=exclude_closed)
    return query


def single_day_query(day, month, year, exclude_closed=True):
    """Returns a MongoDB query returning all posts for a given year."""
    start_date = datetime(year, month, day)
    query = mongo_query(start_date=start_date,
                        end_date=start_date + timedelta(days=10),
                        exclude_closed=exclude_closed)
    return query


def get_mongo_collection(forum):
    """Returns the Mongo collection corresponding to the specified StackExchange forum."""
    client = pymongo.MongoClient()
    posts = client.titlewave[f'{forum}.posts']
    return posts


def get_classifier_dataset_partition(collection, partition, max_size=None):
    ids = get_mongo_ids(collection, {'partition': partition})
    return MongoDataset(collection, ids, classifier_projection, max_size)


def get_summarizer_dataset_partition(collection, partition, max_size=None):
    query = {'partition': partition, 'HasAcceptedAnswer': True}
    ids = get_mongo_ids(collection, query)
    return MongoDataset(collection, ids, summarizer_projection, max_size)

