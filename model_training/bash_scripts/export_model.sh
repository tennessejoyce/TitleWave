#!/bin/bash

# Confirm that we're using the right python interpreter (with tensorflowjs)
which python

cd data/BERT
pytorch_model_loc="BERT_pytorch_model"
keras_model_loc="BERT_keras_model.h5"
tfjs_model_loc="BERT_tfjs_model"
if [ ! -f "$keras_model_loc" ]; then
  echo "Converting from Pytorch to Keras h5..."
  python ../../model_training/python_scripts/pytorch_to_keras.py "$pytorch_model_loc" "$keras_model_loc"
fi
echo "Converting from keras to tensorflowjs..."
#tensorflowjs_converter --input_format keras "$keras_model_loc" "$tfjs_model_loc"