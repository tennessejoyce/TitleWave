import numpy as np
import mongo_dataset
import json
import sys

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
    splits = np.split(array, split_locations).tolist()
    return splits


def main(start_year, end_year, proportions):
    """Select the portion of the dataset to be used, and forms random partitions of specified size."""
    print('Connecting to database...')
    dataset = mongo_dataset.MongoDataset()
    # Clear all
    print('Resetting partitions...')
    dataset.reset_partitions()
    print('Assigning documents to partitions...')
    query = mongo_dataset.year_range_query(start_year, end_year)
    filtered_ids = dataset.get_mongo_ids(query)
    split_ids = split_to_proportions(filtered_ids, list(proportions.values()))
    print('Updating partitions in database...')
    for name, ids in zip(proportions.keys(), split_ids):
        print(f'{name}: {len(ids)} posts')
        dataset.batch_update(ids, {'$set': {'partition': name}})
    print('Indexing partitions...')
    dataset.collection.create_index('partition', sparse=True)


if __name__=='__main__':
    # Partition names and sizes besides the defaults can be specified as a command line argument in JSON format.
    proportions = json.loads(sys.argv[1]) if len(sys.argv) > 1 else default_proportions
    main(proportions)





