import pandas as pd
import os
import sys
import math
import ray
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
from scipy.stats import spearmanr, percentileofscore
from datetime import datetime
from Bio import SeqIO

cwd = os.getcwd()
ray.init()

@ray.remote
def func1():
	os.chdir('./ensemble/CRISPRon')
	os.system('./bin/CRISPRon.sh CRISPRon_data.fasta result')
	os.chdir('..')

@ray.remote
def func2():
	os.chdir('./ensemble/sgDesigner')
	os.system('perl sgDesigner.pl -f sgD_data.fasta')
	os.chdir('..')

@ray.remote
def func3():
	os.chdir('./ensemble/DeepHF')
	os.system('python3 custom_prediction.py')
	os.chdir('..')

@ray.remote
def func4():
	os.chdir('./ensemble/DeepSpCas9')
	os.system('python2 DeepCas9_TestCode.py')
	os.chdir('..')

@ray.remote
def func5():
	os.chdir('./ensemble/TSAM')
	os.system('python2 TSAM_python.py TSAM_data.fasta selfdefined 1 1 0 1')
	os.chdir('..')

def extractGRNA_PAM(string):
	new_string = string[5:-3]
	return new_string

def extract30mer(string):
	new_string = string[1:]
	return new_string

def extract26mer(string):
	new_string = string[5:]
	return new_string

os.chdir(cwd + '/ensemble')
data_file = cwd + '/temp/oligos_v2.0.txt'
data = pd.read_csv(data_file, sep='\t', header=0)
data['gRNA_PAM'] = data.apply(lambda row: extractGRNA_PAM(row['31mer']), axis=1)
data['30mer'] = data.apply(lambda row: extract30mer(row['31mer']), axis=1)
data['26mer'] = data.apply(lambda row: extract26mer(row['31mer']), axis=1)

#############################################################################
############################## Regression ###################################
#############################################################################
CRISPRon_file = cwd + '/ensemble/CRISPRon/CRISPRon_data.fasta'
CRISPRon_tab_file = cwd + '/ensemble/CRISPRon/CRISPRon_data.tab'
sgD_file = cwd + '/ensemble/sgDesigner/sgD_data.fasta'
sgD_tab_file = cwd + '/ensemble/sgDesigner/sgD_data.tab'
DeepHF_file = cwd + '/ensemble/DeepHF/DeepHF_data.txt'
DeepSpCas9_file = cwd + '/ensemble/DeepSpCas9/DeepSpCas9_data.txt'
TSAM_file = cwd + '/ensemble/TSAM/TSAM_data.fasta'
TSAM_tab_file = cwd + '/ensemble/TSAM/TSAM_data.tab'

########################## test data collection #############################
data['gRNA_PAM'].to_csv(DeepHF_file, sep='\t', index=False, header=False)
data['30mer'].to_csv(DeepSpCas9_file, sep='\t', index_label='index', header=True)
data['30mer'].to_csv(CRISPRon_tab_file, sep='\t', index=True, header=False)
SeqIO.convert(CRISPRon_tab_file, 'tab', CRISPRon_file, 'fasta')
os.remove(CRISPRon_tab_file)
data['26mer'].to_csv(sgD_tab_file, sep='\t', index=True, header=False)
SeqIO.convert(sgD_tab_file, 'tab', sgD_file, 'fasta')
os.remove(sgD_tab_file)
data['31mer'].to_csv(TSAM_tab_file, sep='\t', index=True, header=False)
SeqIO.convert(TSAM_tab_file, 'tab', TSAM_file, 'fasta')
os.remove(TSAM_tab_file)
#############################################################################

ray.get([func1.remote(), func2.remote(), func3.remote(), func4.remote(), func5.remote()])

CRISPRon_result_file = cwd + '/ensemble/CRISPRon/result/crispron.csv'
sgD_result_file = cwd + '/ensemble/sgDesigner/result/sgDesigner_V2.0_prediction_result.txt'
DeepHF_result_file = cwd + '/ensemble/DeepHF/DeepHF_results.txt'
DeepSpCas9_result_file = cwd + '/ensemble/DeepSpCas9/RANK_final_DeepCas9_Final.txt'
TSAM_result_file = cwd + '/ensemble/TSAM/predict_results/predict_results.csv'

########################## first layer prediction collection #########################
CRISPRon_df = pd.read_csv(CRISPRon_result_file, sep=',', header=0)
CRISPRon_df.set_index('30mer', inplace=True)
CRISPRon_dict = CRISPRon_df.to_dict()['CRISPRon']

sgD_df = pd.read_csv(sgD_result_file, sep='\t', header=0, index_col=0)
sgD_df['Sequence'] = sgD_df['Sequence'].str.upper()
sgD_df.set_index('Sequence', inplace=True)
sgD_dict = sgD_df.to_dict()['Score']

HF_df = pd.read_csv(DeepHF_result_file, sep='\t', header=0, index_col=0)
HF_df.set_index('gRNA_Seq', inplace=True)
HF_dict = HF_df.to_dict()['Efficiency']

Sp_data = open(DeepSpCas9_result_file, "r")
lines = Sp_data.readlines()
Sp_result = lines[4].strip()[1:-1]
Sp_list = Sp_result.split(', ')
Sp_df = pd.DataFrame(Sp_list, columns=['DeepSpCas9_score'])

TSAM_df = pd.read_csv(TSAM_result_file, sep=',', header=0)
TSAM_df.set_index('spacer', inplace=True)
TSAM_dict = TSAM_df.to_dict()['TSAM_predict']
######################################################################################

os.remove(CRISPRon_file)
os.remove(CRISPRon_result_file)
os.remove(sgD_file)
os.remove(sgD_result_file)
os.remove(DeepHF_file)
os.remove(DeepHF_result_file)
os.remove(DeepSpCas9_file)
os.remove(DeepSpCas9_result_file)
os.remove(TSAM_file)
os.remove(TSAM_result_file)

######################### generate second layer feature set ##########################
second_feature = pd.DataFrame()
second_feature['gRNA'] = data['gRNA']
second_feature['30mer'] = data['30mer']
second_feature['CRISPRon_score'] = second_feature['30mer'].map(CRISPRon_dict)
second_feature['DeepHF_score'] = second_feature['gRNA'].map(HF_dict)
second_feature['DeepSpCas9_score'] = Sp_df['DeepSpCas9_score']
second_feature['sgD_score'] = second_feature['gRNA'].map(sgD_dict)
second_feature['TSAM_score'] = second_feature['gRNA'].map(TSAM_dict)
second_feature = second_feature.drop(columns=['gRNA', '30mer'])
######################################################################################

######################### load model and predict #############################################
model_file = cwd + '/ensemble/model/Ridge_meta_model_ensemble5.joblib'
Ridge_regressor = joblib.load(model_file)
pred = Ridge_regressor.predict(second_feature)
pred_df = pd.DataFrame(pred, columns=['ensemble_prediction'])

def PercentileScore(value):
	percentile_score = percentileofscore(distribution, float(value))
	return math.ceil(percentile_score*10)/10

pred_df['percentile_score'] = pred_df.apply(lambda row: PercentileScore(row['ensemble_prediction']), axis=1)
pred_df = pred_df.drop(columns=['ensemble_prediction'])

pred_file = cwd + '/temp/custom_prediction_result_v2.0.txt'
pred_df.to_csv(pred_file, sep='\t', index=False, header=True)
##############################################################################################

##############################################################################################
##############################################################################################
##############################################################################################





