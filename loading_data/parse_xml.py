import xml.etree.ElementTree as ET
import xmltodict
import json
import pandas as pd
from time import time



name = 'overflow/Posts.xml'
dates = set()
dicts = []
is_first = True
for event, elem in ET.iterparse(name, events=("start", "end")):
	# get the root element
	if is_first:
		root = elem
		is_first = False
	if event=='start':
		continue
	if elem.attrib['PostTypeId']=='1':
			dicts.append(elem.attrib)
	d = elem.attrib['CreationDate'][:4]
	if not (d in dates):
		print(d)
		dates.add(d)
	elem.clear()
	root.clear()
	if len(dicts)>500000:
		df = pd.DataFrame(dicts)
		df.to_csv(f'xml_out/{int(time()*10)}.csv')
		dicts.clear()
		df = df.iloc[0:0]

