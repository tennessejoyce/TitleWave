from transformers import T5ForConditionalGeneration, T5Tokenizer, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from data_loading import get_mongo_dataset, Seq2SeqCollateFn
import os

forum = 'overflow'
year = 2019
os.chdir(forum)
if not os.path.exists('T5'):
    os.mkdir('T5')
os.chdir('T5')

# Set these parameters
batch_size = 16
train_size = 2**16
val_size = 2**14

# Derived parameters
train_steps = train_size // batch_size
steps_between_val = val_size // batch_size

tokenizer = T5Tokenizer.from_pretrained('t5-small')
collate_fn = Seq2SeqCollateFn(inputs_col='Body', outputs_col='Title', tokenizer=tokenizer)

train_dataset, val_dataset = get_mongo_dataset(forum=forum, year=year, mode='t5', val_size=val_size)

# Load the model
print('Loading model...')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


# For now we aren't freezing anything, but it could be smart to freeze the encoder in the first
# part of training. Just try to learn how to write titles first, then improve the encoding after.

def train_model(model, train_dataset, steps):
    train_args = Seq2SeqTrainingArguments(output_dir='checkpoints',
                                          evaluation_strategy='steps',
                                          eval_steps=steps_between_val,
                                          max_steps=steps,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          disable_tqdm=True,
                                          save_steps=2 * steps,
                                          logging_steps=steps_between_val,
                                          fp16=True,
                                          save_total_limit=4)
    trainer = Seq2SeqTrainer(model=model,
                             args=train_args,
                             data_collator=collate_fn,
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset)
    trainer.train()


train_model(model, train_dataset, train_steps)

model.save_pretrained('T5_pytorch_model')
