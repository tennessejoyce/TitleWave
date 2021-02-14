import json

import requests

API_URL = "https://api-inference.huggingface.co/models/tennessejoyce/titlewave-t5-small"


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, data=data)
    return json.loads(response.content.decode("utf-8"))


body = """I'm trying to track accumulation of events over time, e.g. the graphs of total number of COVID Cases & deaths over the past year. My starting data is a list of individuals (rows) with the date for each event in the column. A simplified example would be:
My quick way to count is to compare the whole list of dates for an event (a column)to a single date, e.g. for day 50:
I've been trying to compare each of 3 columns iteratively against each of single dates in my list to build a table I can use to graph accruals (see end of question for example). Any time I try to do this as a vector operation (data.table, apply functions), the result is only one count, not a vector of counts for each date
This seems to compare the vectors of events and dates pairwise, which is the advertised behavior. What I really want is for the whole column to be evaluated against the first element of dates, then the 2nd element of dates, etc. Having used data.table and dpylr for years, I think there should be a more elegant way to do this than looping and counting as I go. The following code works, but I feel I'm missing a simpler, more elegant solution.
Thank you for your suggestions."""

title = 'Counting and counting in a vector operation'

data = query(body)
print(data)
