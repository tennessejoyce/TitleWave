import pymongo
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta


def mongo_query(**kwargs):
    """Create a MongoDB query based on a set of conditions."""
    query = {}
    if 'start_date' in kwargs:
        if not ('CreationDate' in query):
            query['CreationDate'] = {}
        query['CreationDate']['$gte'] = kwargs['start_date']
    if 'end_date' in kwargs:
        if not ('CreationDate' in query):
            query['CreationDate'] = {}
        query['CreationDate']['$lt'] = kwargs['end_date']
    if 'exclude_closed' in kwargs:
        query['Closed'] = kwargs['exclude_closed']
    return query


def year_range_query(start_year, end_year, exclude_closed=True):
    """Returns a MongoDB query returning all posts for a given year."""
    query = mongo_query(start_date=datetime(start_year, 1, 1),
                        end_date=datetime(end_year + 1, 1, 1),
                        exclude_closed=exclude_closed)
    return query


def single_day_query(day, month, year, exclude_closed=True):
    """Returns a MongoDB query returning all posts for a given day."""
    start_date = datetime(year, month, day)
    query = mongo_query(start_date=start_date,
                        end_date=start_date + timedelta(days=10),
                        exclude_closed=exclude_closed)
    return query


class MongoDataset:
    """Interface between MongoDB and the rest of the Python code."""

    def __init__(self, forum='overflow'):
        try:
            client = pymongo.MongoClient()
        except Exception as e:
            message = """Could not connect to MongoDB client. Make sure to start it by executing: 
            sudo systemctl start mongod """
            print(message)
            raise e
        self.collection = client.titlewave[f'{forum}.posts']

    def get_mongo_ids(self, query):
        """Fetches the ids of documents matching a query."""
        result = self.collection.find(query, {'_id': True})
        ids = [row['_id'] for row in result]
        return ids

    def batch_update(self, ids, command, batch_size=256, progress_bar=True):
        """
        Execute an update_many command in batches.

        Parameters:
            ids - The document ids in the Mongo collection of the documents to be updated.
            command - The update command to be executed on each document.
            batch_size - The number of documents to update in a single call of update_many.
            progress_bar - Whether to display a progress bar.
        """
        num_batches = len(ids) // batch_size
        # Split the array into batches of the specified size, and typecast the ids back to Python integers with tolist.
        splits = np.array_split(ids, num_batches).tolist()
        if progress_bar:
            splits = tqdm(splits)
        for batch_ids in splits:
            self.collection.update_many({'_id': {'$in': batch_ids}}, command)

    def get_partition(self, partition, projection):
        """
        Fetches all documents in a specified partition of the dataset.

        Parameters:
            partition - The name of the partition (e.g., "classifier_train")
            projection - Indicates which fields of the documents to return.
        """
        cursor = self.collection.find({'partition': partition}, projection)
        return list(cursor)

    def reset_partitions(self):
        """Remove the partition field from all documents in the collection."""
        self.collection.update_many({'partition': {'$exists': True}}, {'$unset': {'partition': 1}})