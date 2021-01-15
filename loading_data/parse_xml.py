import xml.etree.ElementTree as ET
import pandas as pd
import datetime
import re
from pymongo import MongoClient
import csv
from time import time
#This script parses the raw Stack Overflow data from a single giant >70 GB XML file,
#down into a bunch of csv files that are easier to work with.

def xml_iterator(filename):
	"""
	Iterates through an XMLfile too big to fit into memory, returning a
	dictionary for each element.
	"""
	is_first = True
	for event, elem in ET.iterparse(filename, events=("start", "end")):
		if is_first:
			#Get the root element. We need to clear this after every iteration to avoid memory leak.
			root = elem
			is_first = False
		if event=='start':
			#We are only interested in 'end' events (i.e., after a chunk of data has been read)
			continue
		if elem.attrib:
			yield elem.attrib
		#Clear the data manually to avoid memory leak.
		elem.clear()
		root.clear()

def clean_text(text):
  """
  Remove code blocks, urls, and html tags.
  """
  text = re.sub(r'<code[^>]*>(.+?)</code\s*>', '', text,flags=re.DOTALL | re.MULTILINE)
  text = re.sub(r'<div[^>]*>(.+?)</div\s*>', '', text,flags=re.DOTALL | re.MULTILINE)
  text = re.sub(r'<blockquote[^>]*>(.+?)</blockquote\s*>', '', text,flags=re.DOTALL | re.MULTILINE)
  text = re.sub('<.*?>', '', text)
  text = text.replace('&quot;','"')
  text = re.sub(r'http\S+', '', text)
  text = re.sub(r'www.\S+', '', text)
  return text

def convert_time(s):
	return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")

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
			'Tags': doc['Tags'],
			'AnswerCount': int(doc['AnswerCount']),
			'CommentCount': int(doc['CommentCount']),
			'HasAcceptedAnswer': ('AcceptedAnswerId' in doc),
			'Closed': ('ClosedDate' in doc)
			}


# Name of the XML file, downloaded from https://archive.org/details/stackexchange.
name = 'physics'
#Batch for loading data in chunks
batch_size = 1000

# Metadata 
metadata_cols = ['_id','CreationDate','Score','ViewCount','AnswerCount',
				 'CommentCount', 'HasAcceptedAnswer', 'Closed']

# Access the MongoDB to load the data into.
client = MongoClient()
db = client.titlewave
mongo_posts = db[f'{name}_posts']


filename = f'{name}.stackexchange.com/Posts.xml'
csv_filename = f'{name}_posts.csv'
dates = set()
start_time = time()
i = 0
batch_dicts = []
with open(csv_filename, 'w', newline='') as csv_file:
	csv_writer = csv.DictWriter(csv_file, metadata_cols, extrasaction='ignore')
	csv_writer.writeheader()
	for attrib in xml_iterator(filename):
		if attrib['PostTypeId']!='1':
			# If the post isn't a question, skip it.
			continue
		# Preprocess the features in the dictionary.
		attrib = prune_document(attrib)
		batch_dicts.append(attrib)
		i += 1	
		if i%batch_size == 0:
			# Add to MongoDB
			mongo_posts.insert_many(batch_dicts)
			# Write to a csv (just the metadata features)
			csv_writer.writerows(batch_dicts)
			# Delete the batch to free up memory.
			batch_dicts = []

		# Whenever we get to a new year, print it to indicate progress.
		d = attrib['CreationDate'].year
		if not (d in dates):
			print(d)
			dates.add(d)
	# Write the remaining data
	if batch_dicts:
		mongo_posts.insert_many(batch_dicts)
		csv_writer.writerows(batch_dicts)
print(f'Duration: {time()-start_time:.2f} s')