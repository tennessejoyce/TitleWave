from transformers import BertConfig, TFBertForSequenceClassification
import tensorflow as tf

pytorch_loc = 'BERT_training/final_model'

with tf.device('/CPU:0'):
    config = BertConfig.from_json_file(pytorch_loc+'/config.json')
    model = TFBertForSequenceClassification.from_pretrained(pytorch_loc+'/pytorch_model.bin', from_pt=True, config=config)
    print(model)