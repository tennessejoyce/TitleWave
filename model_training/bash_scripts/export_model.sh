#!/bin/bash

cd data/BERT_training
pytorch_model_loc="BERT_pytorch_model"
keras_model_loc="BERT_keras_model.h5"
tfjs_model_loc="BERT_tfjs_model"
python ../../model_training/python_scripts/pytorch_to_keras.py "$pytorch_model_loc" "$keras_model_loc"
echo "Successfully converted from Pytorch to Keras h5..."
#tensorflowjs_converter --input_format keras "$keras_model_loc" "$tfjs_model_loc"