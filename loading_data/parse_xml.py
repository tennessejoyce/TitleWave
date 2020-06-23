import xml.etree.ElementTree as ET
import xmltodict
import json
import pandas as pd
from time import time

#This script parses the raw Stack Overflow data from a single giant >70 GB XML file,
#down into a bunch of csv files that are easier to work with.

#Name of the XML file, downloaded from https://archive.org/details/stackexchange.
name = 'overflow/Posts.xml'
dates = set()
dicts = []
is_first = True
for event, elem in ET.iterparse(name, events=("start", "end")):
	if is_first:
		#Get the root element. We need to clear this after every iteration to avoid memory leak.
		root = elem
		is_first = False
	if event=='start':
		#We are only interested in 'end' events (i.e., after a chunk of data has been read)
		#Except for the very first event to get the root.
		continue
	if elem.attrib['PostTypeId']=='1':
		#If the post is a question, not an answer, save it to the dictionary.
		dicts.append(elem.attrib)
	#Whenever we get to a new year, print something to indicate progress.
	d = elem.attrib['CreationDate'][:4]
	if not (d in dates):
		print(d)
		dates.add(d)
	#Clear the data manually to avoid memory leak.
	elem.clear()
	root.clear()
	#Once enough questions have been read in, save them to a csv file.
	if len(dicts)>500000:
		df = pd.DataFrame(dicts)
		df.to_csv(f'xml_out/{int(time()*10)}.csv')
		#Clear all of this from memory.
		dicts.clear()
		df = df.iloc[0:0]

