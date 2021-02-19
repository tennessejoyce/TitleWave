import xml.etree.ElementTree as ET
import pandas as pd
import datetime
import re
import pymongo
import csv
from time import time
import sys
import os


# This script parses the raw Stack Overflow data from a single giant >70 GB XML file,
# down into a MongoDB collection and a csv with the metadata.

def xml_iterator(filename):
    """Iterates through an XMLfile too big to fit into memory, returning a dictionary for each element."""
    is_first = True
    for event, elem in ET.iterparse(filename, events=("start", "end")):
        if is_first:
            # Get the root element. We need to clear this after every iteration to avoid memory leak.
            root = elem
            is_first = False
        if event == 'start':
            # We are only interested in 'end' events (i.e., after a chunk of data has been read)
            continue
        if elem.attrib:
            yield elem.attrib
        # Clear the data manually to avoid memory leak.
        elem.clear()
        root.clear()


def clean_text(text):
    """
  Remove code blocks, urls, and html tags.
  """
    text = re.sub(r'<code[^>]*>(.+?)</code\s*>', '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'<div[^>]*>(.+?)</div\s*>', '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'<blockquote[^>]*>(.+?)</blockquote\s*>', '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub('<.*?>', '', text)
    text = text.replace('&quot;', '"')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    return text


def convert_time(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")


def split_tags(s):
    tags = s.split('><')
    tags[0] = tags[0][1:]
    tags[-1] = tags[-1][:-1]
    return tags


def prune_document(doc):
    """
    Takes the dictonary of a post read from XML and takes only certain fields,
    to be placed into the MongoDB.
    """
    return {'_id': int(doc['Id']),
            'CreationDate': convert_time(doc['CreationDate']),
            'Score': int(doc['Score']),
            'ViewCount': int(doc['ViewCount']),
            'Body': clean_text(doc['Body']),
            'Title': doc['Title'],
            'Tags': split_tags(doc['Tags']),
            'AnswerCount': int(doc['AnswerCount']),
            'CommentCount': int(doc['CommentCount']),
            'HasAcceptedAnswer': ('AcceptedAnswerId' in doc),
            'Closed': ('ClosedDate' in doc)
            }


# Name of the forum is passed as a command line argument.
forum = sys.argv[1]
print(f'Forum: {forum}')
os.chdir(forum)

# Batch for loading data in chunks
batch_size = 1000

# Metadata 
metadata_cols = ['_id', 'CreationDate', 'Score', 'ViewCount', 'AnswerCount',
                 'CommentCount', 'HasAcceptedAnswer', 'Closed']

# Access the MongoDB to load the data into.
client = pymongo.MongoClient()
db = client.titlewave
mongo_posts = db[f'{forum}.posts']

# Clear any existing collection.
mongo_posts.drop()

# Parse the xml file into a MongoDB collection and a csv
filename = 'Posts.xml'
csv_filename = 'posts.csv'
print(os.getcwd())
dates = []
start_time = time()
i = 0
i_prev = 0
batch_dicts = []
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, metadata_cols, extrasaction='ignore')
    csv_writer.writeheader()
    for attrib in xml_iterator(filename):
        if attrib['PostTypeId'] != '1':
            # If the post isn't a question, skip it.
            continue
        # Preprocess the features in the dictionary.
        attrib = prune_document(attrib)
        batch_dicts.append(attrib)
        i += 1
        if i % batch_size == 0:
            # Add to MongoDB
            mongo_posts.insert_many(batch_dicts)
            # Write to a csv (just the metadata features)
            csv_writer.writerows(batch_dicts)
            # Delete the batch to free up memory.
            batch_dicts = []

        # Whenever we get to a new year, print it to indicate progress.
        d = attrib['CreationDate'].year
        if not (d in dates):
            posts_loaded = i - i_prev
            duration = time() - start_time
            rate = 10000 * duration / posts_loaded
            print(f'{dates[-1]}: loaded {posts_loaded} posts in {duration:.2f} s ({rate} s per 10,000 posts)')
            dates.append(d)
            i_prev = i
            start_time = time()
    # Write the remaining data
    if batch_dicts:
        mongo_posts.insert_many(batch_dicts)
        csv_writer.writerows(batch_dicts)

print('Creating indices...')
start_time = time()
# Index the MongoDB collection by creation date.
mongo_posts.create_index('CreationDate')
mongo_posts.create_index([('Tags', pymongo.TEXT)])
print(f'Duration: {time() - start_time:.2f} s')
