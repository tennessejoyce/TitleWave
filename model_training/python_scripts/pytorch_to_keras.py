from transformers import BertConfig, TFBertForSequenceClassification
import tensorflow as tf
import sys

if len(sys.argv) < 3:
    print('Please provide names for the input and output files as command line arguments.')
pytorch_loc = sys.argv[1]
tensorflow_loc = sys.argv[2]

input_tensor_size = 384

with tf.device('/CPU:0'):
    config = BertConfig.from_json_file(pytorch_loc+'/config.json')
    model = TFBertForSequenceClassification.from_pretrained(pytorch_loc+'/pytorch_model.bin', from_pt=True, config=config)
    callable = tf.function(model.call)
    concrete_function = callable.get_concrete_function([tf.TensorSpec([None, input_tensor_size], tf.int32, name="input_ids"),
                                                        tf.TensorSpec([None, input_tensor_size], tf.int32, name="attention_mask")])
    tf.saved_model.save(model, tensorflow_loc, signatures=concrete_function)