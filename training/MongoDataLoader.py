import pandas as pd
import pymongo
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split


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

def mongo_dataset(forum, query, projection, sample_size=None):
    """
    Query a MongoDB collection, and convert the results to a pandas Dataframe.

    sample_size can be an int, or a list of ints, in which case multiple dataframe
        will be returns (e.g., for train and test sets).
    """
    client = pymongo.MongoClient()
    posts = client.titlewave[f'{forum}.posts']
    if sample_size:
        result = posts.aggregate([{'$match': query},
                                  {'$sample': {'size': sample_size}},
                                  {'$project': projection}])
    else:
        result = posts.find(query, projection)
    df = pd.DataFrame(result)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    return df

class SequenceClassificationDataset (torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        titles = list(df.Title.values)
        self.encodings = tokenizer(titles, truncation=True, padding=True)
        self.labels = df.Answered.values

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)



def get_title_dataset(forum, tokenizer, sample_size=None, partition_fractions=None):
    query = mongo_query(start_date=datetime(2019, 1, 1),
                        end_date=datetime(2020, 1, 1),
                        exclude_closed=True)
    titles_projection = {'Title': True,
                         'Answered': {'$gt': ['$AnswerCount', 0]},
                         '_id': False}
    df = mongo_dataset('physics', query, titles_projection, sample_size=sample_size)
    dataset = SequenceClassificationDataset(df, tokenizer)
    if partition_fractions:
        partition_sizes = [int(len(dataset)*f) for f in partition_fractions]
        partition_sizes[0] += len(dataset) - sum(partition_sizes)

        return torch.utils.data.random_split(dataset, partition_sizes)
    else:
        return dataset





