from pymongo import MongoClient
import pprint
from datetime import datetime
import sys
import os
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# Name of the forum is passed as a command line argument.
forum = sys.argv[1]
print(f'Forum: {forum}')
os.chdir(forum)

client = MongoClient()
db = client.titlewave
posts = db[f'{forum}.posts']

total_posts = posts.count_documents({})
print(f'{total_posts} posts found...')
if total_posts == 0:
    exit()

print('Counting posts by year...')
start_time = time()
result = posts.aggregate([{'$group': {'_id': {'$year': '$CreationDate'},
                                      'NumPosts': {'$sum': 1},
                                      'AvgViews': {'$avg': '$ViewCount'},
                                      'AnswerProbability': {'$avg': {'$cond': [{'$gt': ['$AnswerCount', 0]}, 1, 0]}}}},
                          {'$sort': {'_id': 1}}
                          ])
result = list(result)
df = pd.DataFrame(result)
print(df)
print(f'Duration: {time() - start_time:.2f} s')

print('Analyzing posts by title length...')
start_time = time()
result = posts.aggregate([{'$group': {'_id': '$TitleChars',
                                      'NumPosts': {'$sum': 1},
                                      'AvgViews': {'$avg': '$ViewCount'},
                                      'AnswerProbability': {'$avg': {'$cond': [{'$gt': ['$AnswerCount', 0]}, 1, 0]}}}},
                          {'$sort': {'_id': 1}}
                          ])
result = list(result)
df = pd.DataFrame(result)
print(df)
print(f'Duration: {time() - start_time:.2f} s')

fig, ax = plt.subplots()
df.plot(x='_id', y='NumPosts', ax = ax, color='blue')
df.plot(x='_id', y='AnswerProbability', ax = ax.twinx(), color='red')
plt.show()
