from pymongo import MongoClient

name = 'physics'

client = MongoClient()
db = client.titlewave
posts = db[f'{name}_posts']

posts.drop()