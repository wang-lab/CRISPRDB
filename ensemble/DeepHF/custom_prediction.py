from prediction_util import *
from training_util import *
import os
import pandas as pd


data_file = './DeepHF_data.txt'

seq = ''

data = open(data_file, "r")
lines = data.readlines()

for line in lines:
	line = line.strip()
	seq = seq + str(line)

result = effciency_predict(seq, 'wt_u6')

result.to_csv('./DeepHF_results.txt', index = None, sep = "\t")
