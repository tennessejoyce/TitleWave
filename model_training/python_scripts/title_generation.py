from transformers import T5ForConditionalGeneration, pipeline
from data_loading import get_t5_dataset, Seq2SeqCollateFn

model = T5ForConditionalGeneration.from_pretrained('T5_pytorch_model')
summarizer = pipeline(task='summarization', model=model, tokenizer='t5-small')

train_dataset, val_dataset = get_t5_dataset('physics', 0.2)

parameters = {'max_length': 20, 'min_length': 4}

for i, row in enumerate(train_dataset):
    if i > 4:
        break
    body = row['Body']
    actual_title = row['Title']
    suggested_title = summarizer(body, **parameters)[0]['summary_text']
    print('Actual title:')
    print(actual_title)
    print('Suggested title:')
    print(suggested_title)
    print('Question body:')
    print(body)
    print()
