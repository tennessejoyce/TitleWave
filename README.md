# TitleWave

## Overview
Stack Overflow is ubiquitous in the programming world as a place where people can ask questions to a community other programmers. In 2019, over 5,000 questions a day were asked, but only 70% of those were answered. To increase your chance of getting an answer it’s really important to have a compelling title so that people actually click on your question. But this can be tough, especially for new users who aren’t familiar with the conventions on the website. To solve this problem, I built TitleWave, a Chrome extension that integrates directly into the Stack Overflow website and helps improve your title.

The algorithm leverages deep learning and natural language processing (NLP) to summarize the key details of your question, and phrases it in a way that is consistent with previously successful questions on similar topics. The improved title quality will make it easier for experts to notice your question, contributing to faster, higher quality answers.

## Install the Chrome extension
I am hoping to make this extension available on the Chrome web store in the near future.
Until then, you can use the development version by following these instructions:
1. Clone this repository (really you just need the chrome_extension folder).
2. In Google Chrome, open the Extension Management page by navigating to chrome://extensions.
3. Enable Developer Mode by clicking the toggle switch in the top right corner.
4. Click the LOAD UNPACKED button and select the chrome_extension folder you downloaded in step 1.
5. Navigate to https://stackoverflow.com/questions/ask and if the extension is working you'll see two new buttons below the title entry box.

## Try out the models on Huggingface
You can also try out the tool without downloading anything by using the Huggingface Inference API. Click [HERE](https://huggingface.co/tennessejoyce/titlewave-bert-base-uncased) for the classification model (gives the probability of getting an answer), and click [HERE](https://huggingface.co/tennessejoyce/titlewave-t5-small) for the summarization model (suggests a title, given the body of the question).

## Retrain the models
If you're interested in retraining the models from scratch, see the Python scripts in the model_training folder.
Here are the steps I took:
1. Download the dataset of Stack Overflow posts from https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z (currently ~16 GB compressed, ~90 GB uncompressed).
2. Preprocess the text (removing HTML tags and codeblocks), and load the dataset into a MongoDB collection (see xml_to_mongo.py).
3. Partition the dataset into train, validation, and test sets for each of the two models (see partition_dataset.py).
4. Fine-tune a classification model, starting from bert-base-uncased (see train_classifier.py)
5. Fine-tune a summarization model, starting from t5-small (see train_summarizer.py)
6. Analyze the performance of these models on a test set (see test_classifier.ipynb and  test_summarizer.ipynb)




