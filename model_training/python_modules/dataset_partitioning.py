import numpy as np
import pymongo
from datetime import datetime

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
    for i in remainder:
        sizes[i] += 1
    if shuffle:
        np.random.shuffle(array)
    split_locations = np.cumsum(sizes)[:-1]
    return np.split(array, split_locations)


def make_partitions(collection, query, proportions=None):
    if proportions is None:
        proportions = default_proportions
    filtered_ids = get_mongo_ids(collection, query)
    split_ids = split_to_proportions(filtered_ids, proportions.values())
    for name, ids in zip(proportions.keys(), split_ids):
        collection.update_many({'_id': {'$in': ids}}, {'$set': {'partition': name}})


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


def get_mongo_collection(forum):
    """Returns the Mongo collection corresponding to the specified StackExchange forum."""
    client = pymongo.MongoClient()
    posts = client.titlewave[f'{forum}.posts']
    return posts
