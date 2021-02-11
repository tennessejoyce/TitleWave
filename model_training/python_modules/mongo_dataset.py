import pymongo
import torch
import numpy as np

# Defines classes to interface MongoDB with Pytorch for data loading.


class MongoDataset(torch.utils.data.Dataset):
    """Wraps a MongoDB collection as a Pytorch Dataset."""

    def __init__(self, collection, indices, projection):
        self.results = list(collection.find({'_id': {'$in': indices}}, projection))

    def __getitem__(self, idx):
        """Retrieve a single document from the MongoDB collection."""
        return self.results[idx]

    def __len__(self):
        return len(self.results)


def split_list(indices, chunk_size):
    """Splits a list into chunks of specified size."""
    # Integers indexing the list of indices.
    meta_indices = np.arange(0, len(indices))
    np.random.shuffle(meta_indices)
    num_chunks = len(indices) // chunk_size
    chunked_meta_indices = np.array_split(meta_indices, num_chunks)
    return [[indices[i] for i in chunk] for chunk in chunked_meta_indices]


class MongoIterableDataset(torch.utils.data.IterableDataset):
    """Wraps a MongoDB collection as a Pytorch IterableDataset which retrieves documents at random."""

    def __init__(self, collection, indices, projection, chunk_size=256):
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
