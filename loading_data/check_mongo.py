from pymongo import MongoClient
import pprint
from datetime import datetime

name = 'physics'

client = MongoClient()
db = client.titlewave
posts = db[f'{name}_posts']

#Search between two dates
for year in range(2010,2021):
	start = datetime(year, 1, 1)
	end = datetime(year+1, 1, 1)
	date_range = {'$gte': start, '$lt': end}
	posts_in_year = posts.count_documents({'CreationDate': date_range})
	answered_posts_in_year = posts.count_documents({'CreationDate': date_range, 'AnswerCount': {'$gt': 0}})
	print(f'{year} {posts_in_year} {answered_posts_in_year/posts_in_year:.1%}')


#posts.drop()