from transformers import T5ForConditionalGeneration, pipeline, \
    BertForSequenceClassification, BertTokenizer, T5Tokenizer
from data_loading import get_mongo_dataset
import numpy as np

forum = 'overflow'
year = 2020

print('Loading BERT...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained(f'{forum}/BERT/BERT_pytorch_model')
print('Loading T5...')
t5_model = T5ForConditionalGeneration.from_pretrained(f'{forum}/T5/T5_pytorch_model')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

print('Loading dataset...')
train_dataset, val_dataset = get_mongo_dataset(forum=forum, year=year, mode='t5', val_size=128)


class TitleGenerator:
    """Combines both models (BERT and T5) to generate good titles."""

    def __init__(self, bert_model, t5_model, bert_tokenizer, t5_tokenizer, **kwargs):
        """Initializes pipelines and hyperparameters."""
        self.classifier = pipeline(task='sentiment-analysis', model=bert_model, tokenizer=bert_tokenizer)
        self.summarizer = pipeline(task='summarization', model=t5_model, tokenizer=t5_tokenizer)
        self.kwargs = kwargs

    def generate_possible_titles(self, prompt):
        results = self.summarizer(prompt, **self.kwargs)
        return [results_dict['summary_text'] for results_dict in results]

    def evaluate_titles(self, titles):
        results = self.classifier(titles)
        return [results_dict['score'] for results_dict in results]

    def choose_best_title(self, prompt):
        titles = self.generate_possible_titles(prompt)
        scores = self.evaluate_titles(titles)
        idx = np.argmax(scores)
        return titles[idx], scores[idx]


hyperparameters = {'max_length': 40,
                   'early_stopping': True,
                   'do_sample': True,
                   'num_return_sequences': 16}

print('Initializing title generator...')
title_generator = TitleGenerator(bert_model, t5_model, bert_tokenizer, t5_tokenizer, **hyperparameters)

print('Generating titles...')
for i, row in enumerate(train_dataset):
    body = row['Body']
    actual_title = row['Title']
    actual_score = title_generator.evaluate_titles(actual_title)[0]
    suggested_title, suggested_score = title_generator.choose_best_title(actual_title)
    print(f'Actual title ({actual_score:.3f}):')
    print(actual_title)
    print(f'Suggested title ({suggested_score:.3f}):')
    print(suggested_title)
    print()
    if i > 4:
        break
