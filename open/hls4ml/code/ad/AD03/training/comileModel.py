import os
import glob
import sys
########################################################################
import pdb
########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common as com
import keras_model

args = com.command_line_chk()

# load parameter.yaml
param = com.yaml_load(args.config)

model = keras_model.get_model(param["model"]["name"], 
                              4*32,
                              hiddenDim=param["model"]["hidden_dim"],
                              encodeDim=param["model"]["encode_dim"],
                              bits=param["model"]["quantization"]["bits"],
                              intBits=param["model"]["quantization"]["int_bits"],
                              reluBits=param["model"]["quantization"]["relu_bits"],
                              reluIntBits=param["model"]["quantization"]["relu_int_bits"],
                              lastBits=param["model"]["quantization"]["last_bits"],
                              lastIntBits=param["model"]["quantization"]["last_int_bits"],
                              l1reg=param["model"]["l1reg"],
                              batchNorm=param["model"]["batch_norm"],
                              halfcode_layers=param["model"]["halfcode_layers"],
                              fan_in_out=param["model"]["fan_in_out"])
model.summary()
model.load_weights('./model/ad03/quantised.h5')
model.summary()
model.save('./model/ad03/model_ToyCar.h5')
