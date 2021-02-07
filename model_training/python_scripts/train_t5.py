from transformers import T5ForConditionalGeneration, T5Tokenizer, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from data_loading import get_t5_dataset, Seq2SeqCollateFn

train_batch_size = 8
val_batch_size = 32

tokenizer = T5Tokenizer.from_pretrained('t5-small')
collate_fn = Seq2SeqCollateFn(inputs_col='Body', outputs_col='Title', tokenizer=tokenizer)

train_dataset, val_dataset = get_t5_dataset('physics', 0.2)

# Load the model
print('Loading model...')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# For now we aren't freezing anything, but it could be smart to freeze the encoder in the first
# part of training. Just try to learn how to write titles first, then improve the encoding after.

def train_model(model, train_dataset):
    train_args = Seq2SeqTrainingArguments(output_dir='checkpoints',
                                          evaluation_strategy='steps',
                                          eval_steps=len(val_dataset) // train_batch_size,
                                          num_train_epochs=1,
                                          per_device_train_batch_size=train_batch_size,
                                          per_device_eval_batch_size=val_batch_size,
                                          disable_tqdm=True,
                                          save_steps=1000000,
                                          fp16=True,
                                          save_total_limit=4)
    trainer = Seq2SeqTrainer(model=model,
                             args=train_args,
                             data_collator=collate_fn,
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset)
    trainer.train()


train_model(model, train_dataset)

model.save_pretrained('T5_pytorch_model')
