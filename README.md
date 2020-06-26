# Stack-Exchange-Title-Generator

## Overview
Stack Overflow is ubiquitous in the programming world as a place where people can ask questions to a community other programmers. In 2019, over 5,000 questions a day were asked, but only 70% of those were answered. To increase your chance of getting an answer it’s really important to have a compelling title so that people actually click on your question. But this can be tough, especially for new users who aren’t familiar with the conventions on the website. To solve this problem, I built TitleWave, a Chrome extension that integrates directly into the Stack Overflow website and helps improve your title. Let’s see how it works.


I'm developing a Chrome plugin that will automatically generate a strong title for your
Stack Exchange question, given the body of text. The algorithm leverages natural language processing
to summarize the key details of your question, and phrases it in a way that is consistent with
previously successful questions on similar topics. The improved title quality will make it easier
for experts to notice your question, contributing to faster, higher quality answers.
Alternatively, the algorithm could be helpful in searching for similar questions, simply by inputing
the suggested title into the existing search function on Stack Exchange.

## Installation
I am hoping to make this extension available on the Chrome web store in the near future.
Until then, you can use the development version by as follows
1. Clone this repository (really you just need the chrome_extension folder).
2. Open the Extension Management page by navigating to chrome://extensions.
3. Enable Developer Mode by clicking the toggle switch next to Developer mode.
4. Click the LOAD UNPACKED button and select the extension directory.
5. Navigate to https://stackoverflow.com/questions/ask and if the extension is worked you'll see two new buttons below the title entry box.

You can also try the webapp version at titlewave.xyz.

## Contents
If you're interested in retraining the model yourself, or otherwise exploring the source code, the folders are organized as follows:
* _chrome\_extension_ contains the javascript for the front-end of the Chrome extension. To runs the actual model, it uses Ajax to send a request to titlewave.xyz, which is hosted on AWS.
* _Flask_ has the python/html code for the webapp, and also has a couple routes used by the Chrome extension only.
* _loading\_data_ contains a single python script that I used to parse the 70 GB XML file downloaded from the Stack Exchange Data Dump (https://archive.org/details/stackexchange) that contains all the post info. I pick out only the questions (not answers) and save these into a bunch of different csv files in chunks of 500,000 questions ordered chronologically.
* _training_ contains two Jupyter notebooks that I ran in Google Colab to finetune BERT and T5 respectively. If you're running these on Colab as well, they will automatically install all the libraries you need (HuggingFace).

## How do I make a Chrome extension like this?
The first step is to get a webapp version running on AWS with Flask, gunicorn, and nginx.
There are good tutorials for that part (https://barnett.science/linux/aws/ansible/2020/05/28/ansible-flask.html).

For developing a standalone Chrome extension, see (https://developer.chrome.com/extensions/getstarted).
If you want to actively modify an existing webpage, you'll probably be writing some Javascript (https://www.w3schools.com/js/).
My workflow was basically click inspect element on the thing in the website I want to modify, figure out its ID so that I can access it in the content script, then look up the javascript command to have it do the thing I want.
For debugging, you can press Ctrl-Shift-J in Chrome to open the console and see error messages etc.

The tricky part was to get these two parts (Flask app and Chrome extension) to talk to each other, which I couldn't find a great tutorial for.
I essentially borrowed my code from a previous Insight fellow who made a Chrome extension (https://github.com/wesbarnett/insight).
Basically there's an AJAX command in the content script you need to modify to send the information back and forth between Flask and Chrome.
This worked fine for me when running the Flask app locally, but when I deployed it on AWS, Chrome was flagging the AJAX request as a security concern because my website was http://titlewave.xyz not https://titlewave.xyz.
To get the HTTPS, you need an SSL certificate, which you can get for free using https://certbot.eff.org/.
Just follow the instructions there, running a few commands on your AWS EC2 instance.
Since HTTPS operates on port 443 (not port 80 as for HTTP) you need to enable that port for your EC2 instance, and you also need to modify the nginx file to recieve on that port (and to use the SSL certificate produced by certbot).




