import pandas as pd
import re

# Regex pattern to match the strings in the raw data
p = re.compile(r'(\d*\.\d)([KMG])(\w*).stackexchange.com.7z download')
# Convert filesize to megabytes
unit_dict = {'K': 1/1000, 'M': 1, 'G': 1000}

def parse_line(line, p):
	"""
	Parses the data out of the raw text line into a dictionary.
	"""
	match = p.match(line)
	if match:
		size, unit, name = match.groups()
		actual_size = float(size) * unit_dict[unit]
		return {'name': name,
				'filesize': actual_size}



filename = 'data_dump_index_raw.txt'
with open(filename, 'r') as f:
	parsed_lines = []
	for line in f:
		d = parse_line(line, p)
		if d:
			parsed_lines.append(d)

df = pd.DataFrame(parsed_lines)
df.to_csv('data_dump_index.csv')

df = df.sort_values('filesize')
df = df.set_index('name')
print(df.head(20))

