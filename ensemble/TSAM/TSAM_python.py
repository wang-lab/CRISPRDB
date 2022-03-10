from __future__ import division
import sys
import numpy as np
import scipy.io as sio
from scipy import stats
import xgboost as xgb
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq
import math
import pickle
from sklearn.metrics import roc_auc_score
import copy
sys.path.insert(0, "/usr/lib64/python2.7/site-packages/libsvm-3.23/python")
from svmutil import *
import csv
import os


model_path='models/'
data_path='data/'
def countGC(s, start, end):
    # 'compute the GC counts and GC contents of a sequences from (start,end)'
    GCcounts = len(s[start:end].replace('A', '').replace('T', ''))
    GCcontent = GCcounts / len(s)
    return GCcounts, GCcontent


def ncf(s, order):
    # nucleotide composition features and position nucletide binary features
    # s--sequence
    # order--1,2,3: A,AA,AAA
    t_list = ["A", "G", "C", "T"]
    L = len(s)
    if order == 1:
        nc = t_list
    elif order == 2:
        nc = [m + n for m in ["A", "G", "C", "T"] for n in ["A", "G", "C", "T"]]
    elif order == 3:
        nc = [m + n + k for m in ["A", "G", "C", "T"] for n in ["A", "G", "C", "T"] for k in ["A", "G", "C", "T"]]
    nc_f = np.zeros((1, 4 ** order))

    pos_fea = np.zeros((1, 1))
    for i in xrange(0, L - order + 1):
        pos = np.zeros((1, 4 ** order))
        for j in xrange(0, len(nc)):
            if s[i:i + order] == nc[j]:
                nc_f[0][j] = nc_f[0][j] + 1
                pos[0][j] = 1
                pos_fea = np.hstack((pos_fea, pos))

    n = len(pos_fea[0])
    pos_fea = pos_fea[0][1:n]
    return nc_f, pos_fea


def evaluate_performance(predict_scores, real, pos_lab, neg_lab, threshold):
    result=np.zeros((7,1))
    predict=np.zeros((len(real),1))
    for i in xrange(0, len(real)):
        if predict_scores[i]>threshold:
            predict[i,0]=1
    result[0,0]=roc_auc_score(real, predict_scores)
    TN = 0;
    TP = 0;
    FP = 0;
    FN = 0;
    for i in xrange(len(predict)):
        # TN p=r=n_l
        if predict[i] == real[i] == neg_lab:
            TN = TN + 1
        elif predict[i] == real[i] == pos_lab:
            TP = TP + 1
        elif predict[i] > real[i]:
            FP = FP + 1
        elif predict[i] < real[i]:
            FN = FN + 1
    result[1, 0] = TN / (TN + FP);
    result[2, 0] = TP / (TP + FN);
    result[3, 0] = TP / (TP + FP);
    result[4, 0] = (TP + TN) / (TP + FP + TN + FN);
    result[5, 0] = 2 * result[2, 0] * result[3, 0] / (result[2, 0] + result[3, 0]);
    result[6, 0] = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))

    return result

def TM_cal(s):
    # computing tm features, s should be 30
    TM_region = np.zeros((1, 4))
    s_20 = s[4:24]
    s_30 = s
    tm_sub1 = mt.Tm_NN(Seq(s_20[2:7]))
    tm_sub2 = mt.Tm_NN(Seq(s_20[7:15]))
    tm_sub3 = mt.Tm_NN(Seq(s_20[15:20]))
    tm_sub4 = mt.Tm_NN(Seq(s_30[0:30]))

    TM_region[0][0] = tm_sub1
    TM_region[0][1] = tm_sub2
    TM_region[0][2] = tm_sub3
    TM_region[0][3] = tm_sub4

    return TM_region


def analysisImprotances(Improtance, featureNum):
    L = len(Improtance)
    keyMatrix = np.zeros((featureNum, L))
    impMatrix = np.zeros((featureNum, L))
    for i in xrange(0, L):
        impro = Improtance[i]
        key = impro.keys()
        for j in xrange(0, len(key)):
            fea_ind = int(key[j][1:len(key[j])])
            fea_imp = impro[key[j]]
            keyMatrix[j, i] = fea_ind
            impMatrix[j, i] = fea_imp

    return keyMatrix, impMatrix


def binary_seqs(obs_seqs, t):
    nts = ['A', 'C', 'G', 'T']
    N = len(obs_seqs)
    n = len(obs_seqs[0])
    nc = []
    if t == 1:
        nc = nts
    elif t == 2:
        nc = [m + l for m in nts for l in nts]

    alpStat = np.zeros((N, 1 + n - t))
    for i in xrange(0, N):
        seq = obs_seqs[i]
        for j in xrange(0, 1 + n - t):
            mer = seq[j:j + t]
            ind = nc.index(mer)
            alpStat[i, j] = ind

    return alpStat


def TranEmis(alpStat, t, au, ad):
    N = len(alpStat[:, 0])
    n = len(alpStat[0, :])
    b = np.zeros((1, n + 2))
    c = np.zeros((n + 1, 1))
    d = np.eye(n + 1)
    Tr = np.hstack((c, d))
    Tr = np.vstack((Tr, b))
    tr = Tr.copy()
    tr[tr == 0] = float('-inf')
    tr[tr == 1] = 0

    Et = np.zeros((n + 2, 4 ** t))
    et = Et.copy()
    et[et == 0] = float('-inf')
    for i in xrange(1, n + 1):
        for j in xrange(0, 4 ** t):
            count = np.where(alpStat[:, i - 1] == j)
            p = (len(count[0]) + au) / (N + ad)
            if p == 0:
                et[i, j] = float('-inf')
            else:
                et[i, j] = math.log(p)

    return tr, et


def viterbi(seq, logTR, logE):
    numStates = len(logTR[:, 0])
    L = len(seq)

    v = np.zeros((numStates, 1))
    v[v == 0] = float('-inf')
    v[0, 0] = 0

    for count in xrange(0, L):
        scount = int(seq[count])
        vOld = copy.deepcopy(v)
        for state in xrange(0, numStates):
            bestVal = float('-inf')
            bestPTR = 0
            for inner in xrange(0, numStates):
                val = vOld[inner][0]
                if val > bestVal:
                    bestVal = val
                    bestPTR = inner
            v[state][0] = logE[state, scount] + bestVal + logTR[bestPTR, state]
            # vOld = copy.deepcopy(v)
    logp = max(v)
    return logp[0]


def multi_pHMMs(alpStat_tr, train_score, group, t, au, ad):
    max_score = max(train_score)
    step = 0
    if max_score > 1:
        step = 100 / group
    else:
        step = 1 / group

    scores0 = train_score.copy()
    scores0[(scores0 >= 0) & (scores0 <= step)] = math.e
    index0 = np.where(scores0 == math.e)
    seq0 = alpStat_tr[index0[0], :]
    Tr = []
    Et = []
    tr0, et0 = TranEmis(seq0, t, au, ad)
    Tr.append(tr0)
    Et.append(et0)

    for i in xrange(2, group + 1):
        scores = train_score.copy()
        scores[(scores > (i - 1) * step) & (scores <= i * step)] = math.e
        index = np.where(scores == math.e)
        seq = alpStat_tr[index[0], :]
        tr, et = TranEmis(seq, t, au, ad)
        Tr.append(tr)
        Et.append(et)
    return Tr, Et


def pHMM_fea(Tr, Et, alpStat, group, t, au, ad):
    # tr, et=TranEmis(test_seq, t, au, ad)
    n = len(alpStat[:, 0])
    lgps = np.zeros((n, group))
    for i in xrange(0, n):
        seq = alpStat[i, :]
        for j in xrange(0, group):
            tr_t = copy.deepcopy(Tr[j])
            et_t = copy.deepcopy(Et[j])
            # t_s=time.time()
            logp = viterbi(seq, tr_t, et_t)
            # t_e=time.time()
            # print t_e-t_s
            lgps[i, j] = logp

    return lgps


def pHMM_fea_all(alpStat_tr, train_score, alpStat_te, t, group, au, ad):
    Tr, Et = multi_pHMMs(alpStat_tr, train_score, group, t, au, ad)
    # alpStat_tr=binary_seqs(train_seq, t)
    # alpStat_te=binary_seqs(test_seq, t)
    lgps_tr = pHMM_fea(Tr, Et, alpStat_tr, group, t, au, ad)
    lgps_te = pHMM_fea(Tr, Et, alpStat_te, group, t, au, ad)

    return lgps_tr, lgps_te


def cross_validation_xg(fea_apply, scores, splits, lam, b_round, md, corr_type):
    split_num = len(splits[0])
    Improtance = []
    Spr = []
    Pre_score = np.zeros((2, 1))

    for spl in xrange(0, split_num):
        Split = splits[:, spl]
        train_index = np.where(Split == 1)
        test_index = np.where(Split == 0)
        x_train = fea_apply[train_index]
        x_test = fea_apply[test_index]
        y_train = scores[train_index]
        y_test = scores[test_index]

        data_train = xgb.DMatrix(x_train, y_train)
        data_test = xgb.DMatrix(x_test, y_test)

        param = {'lambda': lam, 'max_depth': md, 'objective': 'reg:tweedie', 'nthread': 8, 'booster': 'gbtree',
                 'silent': 1}
        bst = xgb.train(param, data_train, num_boost_round=b_round)  # evals=watch_list)
        importance = bst.get_fscore()
        Improtance.append(importance)
        y_pred = bst.predict(data_test)
        Y_pred = np.zeros((len(y_pred), 1))

        for i in xrange(0, len(y_pred)):
            Y_pred[i][0] = y_pred[i]

        if corr_type == 1:
            spr = stats.spearmanr(y_test, y_pred)
        else:
            spr = stats.pearsonr(y_test, Y_pred)

        Spr.append(spr[0])
        Pre_score = np.vstack((Pre_score, Y_pred))

    Pre_scores = Pre_score[2:len(Pre_score)]
    return Improtance, Spr, Pre_scores


def mapminmax(feature, ymax, ymin):
    minmax = np.zeros((2, len(feature[0, :])))
    minmax[0][0] = max(feature[:, 0])
    minmax[1][0] = min(feature[:, 0])
    norm_fea = (ymax - ymin) * (feature[:, 0] - min(feature[:, 0])) / (max(feature[:, 0]) - min(feature[:, 0])) + ymin
    for i in xrange(1, len(feature[0, :])):
        minmax[0][i] = max(feature[:, i])
        minmax[1][i] = min(feature[:, i])
        norm = (ymax - ymin) * (feature[:, i] - min(feature[:, i])) / (max(feature[:, i]) - min(feature[:, i])) + ymin
        norm_fea = np.column_stack((norm_fea, norm))
    return minmax, norm_fea

def output_matrix(matrix,name):
    file_path = name  + '.csv'
    csv_file=open(file_path, 'wb')
    writer = csv.writer(csv_file, delimiter=',')
    for line in matrix:
        writer.writerow(line)

def libsvm_dataformat(feature, score, flag, data_index, pretype, featype):
    if flag == 'train':
        #output_matrix(feature, 'train_ori')
        minmax, norm_fea = mapminmax(feature, 1, 0)
        #output_matrix(norm_fea, 'train_norm')
        pickle.dump(minmax, open(model_path + 'minmax' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))

        file_path = model_path + 'train_fea_apply' + str(data_index) + '_' + str(pretype) + '_' + str(featype)
    elif flag == 'test':
        #output_matrix(feature, 'test_ori')
        # pickle.dump(Fea_index, open('minmax', 'wb'))
        ymax = 1
        ymin = 0
        minmax = pickle.load(open(model_path + 'minmax' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))

        norm_fea = (ymax - ymin) * (feature[:, 0] - minmax[1, 0]) / (minmax[0, 0] - minmax[1, 0]) + ymin
        for i in xrange(1, len(feature[0, :])):
            norm = (ymax - ymin) * (feature[:, i] - minmax[1, i]) / (minmax[0, i] - minmax[1, i]) + ymin
            norm_fea = np.column_stack((norm_fea, norm))

        file_path = model_path + 'test_fea_apply' + str(data_index) + '_' + str(pretype) + '_' + str(featype)
        #output_matrix(norm_fea, 'test_norm')

    f = open(file_path, 'w')
    rows = len(score)
    columns = len(norm_fea[0, :])
    for i in xrange(0, rows):
        f.write(str(score[i, 0]) + '\t')
        for j in xrange(0, columns):
            f.write(str(j + 1) + ':' + str(norm_fea[i, j]) + '\t')
        f.write('\n')
    f.close()
    return file_path


def cross_validation_cls_xg(fea_apply, labels, splits, eta, b_round, md):
    index_p = np.where(labels == 1)
    nump = len(index_p[0])
    Fea_p = fea_apply[0:nump, :]
    Fea_n = fea_apply[nump:len(labels), :]

    lab_p = labels[0:nump, :]
    lab_n = labels[nump:len(labels), :]

    split_num = len(np.unique(splits[:, 0]))
    runNum = len(splits[0, :])
    Improtance = []
    Mean_auc = []
    Pre_score = np.zeros((len(labels), 1))
    for run in xrange(0, runNum):
        split = splits[:, run]
        split_p = split[0:len(lab_p)]
        split_n = split[len(lab_p):(len(lab_p) + len(lab_n))]
        pre_score = np.zeros((2, 1))

        auc = []
        for spl in xrange(0, split_num):
            train_p_ind = np.where(split_p != (spl + 1))
            train_n_ind = np.where(split_n != (spl + 1))

            test_p_ind = np.where(split_p == (spl + 1))
            test_n_ind = np.where(split_n == (spl + 1))

            train_p_fea = Fea_p[train_p_ind[0], :]
            train_n_fea = Fea_n[train_n_ind[0], :]

            test_p_fea = Fea_p[test_p_ind[0], :]
            test_n_fea = Fea_n[test_n_ind[0], :]

            train_p_lab = lab_p[train_p_ind[0], :]
            train_n_lab = lab_n[train_n_ind[0], :]

            test_p_lab = lab_p[test_p_ind[0], :]
            test_n_lab = lab_n[test_n_ind[0], :]

            x_train = np.vstack((train_p_fea, train_n_fea))
            x_test = np.vstack((test_p_fea, test_n_fea))
            y_train = np.vstack((train_p_lab, train_n_lab))
            y_test = np.vstack((test_p_lab, test_n_lab))

            data_train = xgb.DMatrix(x_train, label=y_train)
            data_test = xgb.DMatrix(x_test, label=y_test)

            # watch_list = [(data_test, 'eval'), (data_train, 'train')]
            param = {'max_depth': md, 'eta': eta, 'silent': 1, 'objective': 'binary:logistic', 'booster': 'gbtree'}

            bst = xgb.train(param, data_train, num_boost_round=b_round)  # , evals=watch_list)
            y_hat = bst.predict(data_test)
            Y_hat = np.zeros((len(y_hat), 1))
            for y in range(0, len(y_hat)):
                Y_hat[y][0] = y_hat[y]
            importance = bst.get_fscore()
            Improtance.append(importance)

            pre_score = np.vstack((pre_score, Y_hat))
            AUC = roc_auc_score(y_test, Y_hat)
            auc.append(AUC)

        Pre_score = np.hstack((Pre_score, pre_score[2:(len(labels) + 2), :]))
        mean_auc = np.mean(auc)
        Mean_auc.append(mean_auc)
    Mean_AUC = np.mean(Mean_auc)

    return Improtance, Mean_AUC, Pre_score[:, 1:len(Pre_score[0, :])]


def cross_validation_cls_libsvm(train_seqs, Importance, featureAll, labels, splits, C, G, percent, au, ad):
    keyMatrix, impMatrix = analysisImprotances(Importance, len(featureAll[0, :]))
    runnum = len(keyMatrix[0, :])
    fea_index = []
    for i in xrange(0, runnum):
        imp = np.column_stack((keyMatrix[:, i], impMatrix[:, i]))

        Imp = sorted(0 - imp, key=lambda row: row[1])
        for j in xrange(0, percent):
            fea_index.append(int(0 - Imp[j][0]))
    Fea_index = np.unique((fea_index)).tolist()
    pickle.dump(Fea_index, open(model_path + 'Fea_index_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))
    fea_apply = featureAll[:, Fea_index]

    runNum = len(splits[0, :])
    split_num = len(np.unique(splits[:, 0]))
    Pre_scores = np.zeros((len(labels), 1))
    Mean_auc = []
    para = '-t 2 -b 1' + ' -c ' + str(2 ** C) + ' -g ' + str(2 ** G)
    index_p = np.where(labels == 1)
    nump = len(index_p[0])
    fea_p = fea_apply[0:nump, :]
    fea_n = fea_apply[nump:len(labels), :]
    lab_p = labels[0:nump, :]
    lab_n = labels[nump:len(labels), :]
    seqs_p = train_seqs[0:nump, :]
    seqs_n = train_seqs[nump:len(labels), :]
    for run in xrange(0, runNum):
        split = splits[:, run]
        split_p = split[0:nump]
        split_n = split[nump:len(labels)]

        Pre_score = np.zeros((2, 1))

        auc = []
        for spl in xrange(0, split_num):
            train_p_ind = np.where(split_p != (spl + 1))
            train_n_ind = np.where(split_n != (spl + 1))

            test_p_ind = np.where(split_p == (spl + 1))
            test_n_ind = np.where(split_n == (spl + 1))

            train_p_fea = fea_p[train_p_ind[0], :]
            train_n_fea = fea_n[train_n_ind[0], :]

            test_p_fea = fea_p[test_p_ind[0], :]
            test_n_fea = fea_n[test_n_ind[0], :]

            train_p_lab = lab_p[train_p_ind[0], :]
            train_n_lab = lab_n[train_n_ind[0], :]

            test_p_lab = lab_p[test_p_ind[0], :]
            test_n_lab = lab_n[test_n_ind[0], :]

            train_seq_p = seqs_p[train_p_ind[0], :]
            train_seq_n = seqs_n[train_n_ind[0], :]

            test_seq_p = seqs_p[test_p_ind[0], :]
            test_seq_n = seqs_n[test_n_ind[0], :]

            train_seqs = np.vstack((train_seq_p, train_seq_n))
            test_seqs = np.vstack((test_seq_p, test_seq_n))
            train_label = np.vstack((train_p_lab, train_n_lab))
            label = np.vstack((train_p_lab, train_n_lab, test_p_lab, test_n_lab))

            TRs, ETs, scores_tr = pHmmGE(train_seqs, train_label, train_seqs, au, ad)
            TRs, ETs, scores_te = pHmmGE(train_seqs, train_label, test_seqs, au, ad)
            fea_seq = np.vstack((train_p_fea, train_n_fea, test_p_fea, test_n_fea))
            scores = np.vstack((scores_tr, scores_te))
            feas = np.hstack((scores, fea_seq))

            file_path = libsvm_dataformat(feas, label, 'train', data_index, pretype, featype)
            Sco, Fea = svm_read_problem(file_path)

            x_train = Fea[0:len(train_p_lab) + len(train_n_lab)]
            x_test = Fea[len(train_p_lab) + len(train_n_lab):len(Sco)]
            y_train = Sco[0:len(train_p_lab) + len(train_n_lab)]
            y_test = Sco[len(train_p_lab) + len(train_n_lab):len(Sco)]

            m = svm_train(y_train, x_train, para)
            p_label_a, p_acc, p_val = svm_predict(y_test, x_test, m, '-b 1')
            Y_pred = np.zeros((len(p_label_a), 1))
            for i in xrange(0, len(p_label_a)):
                Y_pred[i][0] = p_val[i][0]

            Pre_score = np.vstack((Pre_score, Y_pred))
            AUC = roc_auc_score(y_test, Y_pred)
            auc.append(AUC)

        Pre_scores = np.hstack((Pre_scores, Pre_score[2:len(Pre_score), :]))
        mean_auc = np.mean(auc)
        Mean_auc.append(mean_auc)
    Mean_AUC = np.mean(Mean_auc)
    return Mean_AUC, Pre_scores[:, 1:len(Pre_scores[0, :])], Fea_index


def cross_validation_libsvm(train_seqs, Importance, featureAll, scores, splits, C, G, p, percent, corr_type, group, au,
                            ad):
    keyMatrix, impMatrix = analysisImprotances(Importance, len(featureAll[0, :]))
    runnum = len(keyMatrix[0, :])
    fea_index = []
    for i in xrange(0, runnum):
        imp = np.column_stack((keyMatrix[:, i], impMatrix[:, i]))
        Imp = sorted(0 - imp, key=lambda row: row[1])
        for j in xrange(0, percent):
            fea_index.append(int(0 - Imp[j][0]))
    Fea_index = np.unique((fea_index)).tolist()
    pickle.dump(Fea_index, open(model_path + 'Fea_index_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))
    fea_apply = featureAll[:, Fea_index]

    alpStat_tr = train_seqs

    split_num = len(splits[0])
    Spr = []
    Pre_score = np.zeros((2, 1))
    para = '-s 3 -t 2 -p ' + str(p) + ' -c ' + str(2 ** C) + ' -g ' + str(2 ** G)

    for spl in xrange(0, split_num):
        Split = splits[:, spl]
        train_index = np.where(Split == 1)
        test_index = np.where(Split == 0)

        train_seq = alpStat_tr[train_index[0], :]
        test_seq = alpStat_tr[test_index[0], :]
        train_score = scores[train_index[0], :]
        test_score = scores[test_index[0], :]
        lgps_tr1, lgps_te1 = pHMM_fea_all(train_seq[:, 0:20], train_score, test_seq[:, 0:20], 1, group, au, ad)
        lgps_tr2, lgps_te2 = pHMM_fea_all(train_seq[:, 20:39], train_score, test_seq[:, 20:39], 2, group, au, ad)
        phmm_fea_tr = np.hstack((lgps_tr1, lgps_tr2))
        phmm_fea_te = np.hstack((lgps_te1, lgps_te2))
        phmm_fea = np.vstack((phmm_fea_tr, phmm_fea_te))

        seq_fea_tr = fea_apply[train_index[0], :]
        seq_fea_te = fea_apply[test_index[0], :]

        seq_fea = np.vstack((seq_fea_tr, seq_fea_te))

        fea_all = np.hstack((seq_fea, phmm_fea))
        scores_all = np.row_stack((train_score, test_score))

        file_path = libsvm_dataformat(fea_all, scores_all, 'train', data_index, pretype, featype)
        Sco, Fea = svm_read_problem(file_path)

        xtrain = Fea[0:len(train_score)]
        xtest = Fea[len(train_score):len(scores)]
        ytrain = Sco[0:len(train_score)]
        ytest = Sco[len(train_score):len(scores)]

        m = svm_train(ytrain, xtrain, para)
        p_label_a, p_acc, p_val = svm_predict(ytest, xtest, m)
        Y_pred = np.zeros((len(p_label_a), 1))
        for i in xrange(0, len(p_label_a)):
            Y_pred[i][0] = p_label_a[i]

        if corr_type == 1:
            spr = stats.spearmanr(ytest, p_label_a)
            print spr
        else:
            spr = stats.pearsonr(ytest, p_label_a)

        Pre_score = np.vstack((Pre_score, Y_pred))
        Spr.append(spr[0])

    Pre_scores = Pre_score[2:len(Pre_score)]
    return Spr, Pre_scores, Fea_index


def cross_validation_cls_ens(splits, labels, pre_score1, pre_score2, k):
    runNum = len(splits[0, :])
    split_num = len(np.unique(splits[:, 0]))
    auc = np.zeros((2, 3))
    Pre_Scores = np.zeros((len(labels), runNum))
    for j in xrange(0, runNum):
        for i in xrange(0, len(labels), 1):
            Pre_Scores[i][j] = k * pre_score1[i, j] + (1 - k) * pre_score2[i, j]
    for run in xrange(0, runNum):
        split = splits[:, run]
        t = 0
        for spl in xrange(0, split_num):
            test_index = np.where(split == spl + 1)
            test_label = labels[test_index[0], :]
            pre_score_xg = pre_score1[t:t + len(test_index[0]), run]
            pre_score_svm = pre_score2[t:t + len(test_index[0]), run]
            pre_score_ens = Pre_Scores[t:t + len(test_index[0]), run]
            AUCs = np.zeros((1, 3))
            AUCs[0, 0] = roc_auc_score(test_label, pre_score_xg)
            AUCs[0, 1] = roc_auc_score(test_label, pre_score_svm)
            AUCs[0, 2] = roc_auc_score(test_label, pre_score_ens)
            # print AUCs
            auc = np.vstack((auc, AUCs))
            t = t + len(test_index[0])

    AUC = auc[2:len(auc), :]
    Pre_score = np.hstack((pre_score1, pre_score2, Pre_Scores))
    return AUC, Pre_score, Pre_Scores


def cross_validation_ens(splits, scores, pre_score1, pre_score2, corr_type, k):
    split_num = len(splits[0])
    t = 0
    Spr = np.zeros((2, 3))
    Pre_scores = np.zeros((len(pre_score1), 3))
    for i in xrange(0, len(pre_score1)):
        Pre_scores[i][0] = pre_score1[i, 0]
        Pre_scores[i][1] = pre_score2[i, 0]
        Pre_scores[i][2] = k * pre_score1[i, 0] + (1 - k) * pre_score2[i, 0]

    for spl in xrange(0, split_num):
        Split = splits[:, spl]
        test_index = np.where(Split == 0)
        pre_score_xg = Pre_scores[t:t + len(test_index[0]), 0]
        pre_score_svm = Pre_scores[t:t + len(test_index[0]), 1]
        pre_score_ens = Pre_scores[t:t + len(test_index[0]), 2]
        y_test = scores[test_index[0],0]
        sprs = np.zeros((1, 3))
        if corr_type == 1:
            spr_xg = stats.spearmanr(y_test, pre_score_xg)
            spr_svm = stats.spearmanr(y_test, pre_score_svm)
            spr_ens = stats.spearmanr(y_test, pre_score_ens)
            sprs[0, 0] = spr_xg[0]
            sprs[0, 1] = spr_svm[0]
            sprs[0, 2] = spr_ens[0]
        else:
            spr_xg = stats.pearsonr(y_test, pre_score_xg)
            spr_svm = stats.pearsonr(y_test, pre_score_svm)
            spr_ens = stats.pearsonr(y_test, pre_score_ens)
            sprs[0, 0] = spr_xg[0]
            sprs[0, 1] = spr_svm[0]
            sprs[0, 2] = spr_ens[0]
        Spr = np.vstack((Spr, sprs))

        t = t + len(test_index[0])
    Sprs = Spr[2:len(Spr), :]
    return Sprs, Pre_scores


def phmm_scores(test_seqs, TRs, ETs):
    scores = np.zeros((len(test_seqs[:, 0]), 2))
    for i in xrange(0, len(test_seqs[:, 0])):
        Tr1_p = TRs['Tr1_p']
        Tr1_n = TRs['Tr1_n']
        Tr2_p = TRs['Tr2_p']
        Tr2_n = TRs['Tr2_n']
        Et1_p = ETs['Et1_p']
        Et1_n = ETs['Et1_n']
        Et2_p = ETs['Et2_p']
        Et2_n = ETs['Et2_n']

        lp1_p = viterbi(test_seqs[i, 0:20], Tr1_p, Et1_p)
        lp1_n = viterbi(test_seqs[i, 0:20], Tr1_n, Et1_n)

        lp2_p = viterbi(test_seqs[i, 20:39], Tr2_p, Et2_p)
        lp2_n = viterbi(test_seqs[i, 20:39], Tr2_n, Et2_n)

        if lp1_p != 0:
            scores[i][0] = lp1_n / lp1_p
        else:
            scores[i][0] = 0

        if lp2_p != 0:
            scores[i][1] = lp2_n / lp2_p
        else:
            scores[i][1] = 0

    return scores


def pHmmGE(train_seqs, labels, test_seqs, au, ad):
    index_p = np.where(labels == 1)
    nump = len(index_p[0])
    TRs = {}
    ETs = {}
    Tr1_p, Et1_p = TranEmis(train_seqs[0:nump, 0:20], 1, au, ad)
    Tr1_n, Et1_n = TranEmis(train_seqs[nump:len(labels), 0:20], 1, au, ad)
    Tr2_p, Et2_p = TranEmis(train_seqs[0:nump, 20:39], 2, au, ad)
    Tr2_n, Et2_n = TranEmis(train_seqs[nump:len(labels), 20:39], 2, au, ad)
    TRs.update({'Tr1_p': Tr1_p, 'Tr2_p': Tr2_p, 'Tr1_n': Tr1_n, 'Tr2_n': Tr2_n})
    ETs.update({'Et1_p': Et1_p, 'Et2_p': Et2_p, 'Et1_n': Et1_n, 'Et2_n': Et2_n})

    scores = phmm_scores(test_seqs, TRs, ETs)

    return TRs, ETs, scores


def train_model_classification(train_seqs, train_fea, labels, data_index, pretype, featype, phmmtype, eta, md, b_round,
                               p, C, G, au, ad):
    x_train = train_fea
    y_train = labels
    data_train = xgb.DMatrix(x_train, y_train)

    param = {'max_depth': md, 'eta': eta, 'silent': 1, 'objective': 'binary:logistic', 'booster': 'gbtree'}
    bst = xgb.train(param, data_train, num_boost_round=b_round)
    filename = model_path + 'trained_final_xg_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '.model'
    bst.save_model(filename)

    if data_index == 6 or data_index == 7 or data_index == 8 or data_index == 9:
        Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(5) + '_' + str(pretype) + '_' + str(featype), 'rb'))
    else:
        Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
    fea_apply = train_fea[:, Fea_index]

    if phmmtype == 1:
        TRs, ETs, scores = pHmmGE(train_seqs, labels, train_seqs, au, ad)
        pickle.dump(TRs, open(model_path + 'trained_final_TRs_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))
        pickle.dump(ETs, open(model_path + 'trained_final_ETs_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))

        fea_all = np.hstack((scores, fea_apply))
    elif phmmtype == 0:
        fea_all = fea_apply

    file_path = libsvm_dataformat(fea_all, labels, 'train', data_index, pretype, featype)
    Sco, Fea = svm_read_problem(file_path)
    para = '-t 2 -b 1 -p 0.14' + ' -c ' + str(2 ** C) + ' -g ' + str(2 ** G)
    m = svm_train(Sco, Fea, para)
    svm_save_model(model_path + 
        'trained_final_libsvm_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(phmmtype) + '.model', m)

    return data_index, pretype, featype, phmmtype


def train_model_regression(train_seqs, train_fea, scores, data_index, pretype, featype, phmmtype, lam, md, b_round, p,
                           C, G, group, au, ad):
    x_train = train_fea
    y_train = scores

    data_train = xgb.DMatrix(x_train, y_train)

    param = {'lambda': lam, 'max_depth': md, 'objective': 'reg:tweedie', 'nthread': 8, 'booster': 'gbtree', 'silent': 1}
    bst = xgb.train(param, data_train, num_boost_round=b_round)  # evals=watch_list)
    filename = model_path + 'trained_final_xg_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '.model'
    bst.save_model(filename)

    Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
    fea_apply = train_fea[:, Fea_index]

    if phmmtype == 1:
        Tr1, Et1 = multi_pHMMs(train_seqs[:, 0:20], scores, group, 1, au, ad)
        Tr2, Et2 = multi_pHMMs(train_seqs[:, 20:39], scores, group, 2, au, ad)
        lgps_tr1 = pHMM_fea(Tr1, Et1, train_seqs[:, 0:20], group, 1, au, ad)
        lgps_tr2 = pHMM_fea(Tr2, Et2, train_seqs[:, 20:39], group, 2, au, ad)
        pickle.dump(Tr1, open(model_path + 'trained_final_Tr1_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))
        pickle.dump(Tr2, open(model_path + 'trained_final_Tr2_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))
        pickle.dump(Et1, open(model_path + 'trained_final_Et1_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))
        pickle.dump(Et2, open(model_path + 'trained_final_Et2_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'wb'))
        fea_all = np.hstack((fea_apply, lgps_tr1, lgps_tr2))
    elif phmmtype == 0:
        fea_all = fea_apply
    file_path = libsvm_dataformat(fea_all, scores, 'train', data_index, pretype, featype)
    Sco, Fea = svm_read_problem(file_path)
    para = '-s 3 -t 2 -p ' + str(p) + ' -c ' + str(2 ** C) + ' -g ' + str(2 ** G)
    m = svm_train(Sco, Fea, para)
    svm_save_model(model_path + 
        'trained_final_libsvm_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(phmmtype) + '.model', m)

    return data_index, pretype, featype, phmmtype


def test_models_classification(test_fea, test_seqs, test_labels, data_index, pretype, featype, phmmtype, au, ad, k):
    modelfile = model_path + 'trained_final_xg_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '.model'
    Pre_scores = np.zeros((len(test_labels), 3))
    # loaded_model = pickle.load(open(modelfile, 'rb'))
    bst = xgb.Booster({'nthread': 8})  # init model
    bst.load_model(modelfile)  # load data
    x_test = test_fea
    y_test = test_labels
    data_test = xgb.DMatrix(x_test, y_test)
    y_pred = bst.predict(data_test)

    for i in xrange(0, len(y_pred)):
        if y_pred[i]>1:
            Pre_scores[i][0] = 1
        elif y_pred[i]<0:
            Pre_scores[i][0] = 0
        else:
            Pre_scores[i][0] = y_pred[i]

    if data_index == 6 or data_index == 7 or data_index == 8 or data_index == 9:
        Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(5) + '_' + str(pretype) + '_' + str(featype), 'rb'))
    else:
        Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
    fea_apply = test_fea[:, Fea_index]

    if phmmtype == 1:
        TRs = pickle.load(open(model_path + 'trained_final_TRs_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        ETs = pickle.load(open(model_path + 'trained_final_ETs_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        scores = phmm_scores(test_seqs, TRs, ETs)

        fea_all = np.hstack((scores, fea_apply))
    elif phmmtype == 0:
        fea_all = fea_apply

    file_path = libsvm_dataformat(fea_all, test_labels, 'test', data_index, pretype, featype)
    Sco, Fea = svm_read_problem(file_path)
    m = svm_load_model(model_path + 
        'trained_final_libsvm_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(phmmtype) + '.model')
    y_test = Sco
    fea_test = Fea
    p_label_a, p_acc, p_val = svm_predict(y_test, fea_test, m, '-b 1')
    for i in xrange(0, len(p_label_a)):
        if p_val[i][0] > 1:
            Pre_scores[i][1] = 1
        elif p_val[i][0] < 0:
            Pre_scores[i][1] = 0
        else:
            Pre_scores[i][1] = p_val[i][0]
        Pre_scores[i][2] = k * Pre_scores[i][0] + (1 - k) * Pre_scores[i][1]

    return Pre_scores


def test_models_regression(test_fea, test_seqs, test_scores, data_index, pretype, featype, phmmtype, group, au, ad, k):
    ###xgboost
    Pre_scores = np.zeros((len(test_scores), 3))
    modelfile = model_path + 'trained_final_xg_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '.model'
    # loaded_model = pickle.load(open(modelfile, 'rb'))

    bst = xgb.Booster({'nthread': 8})  # init model
    bst.load_model(modelfile)  # load data
    x_test = test_fea
    y_test = test_scores
    data_test = xgb.DMatrix(x_test, y_test)
    y_pred = bst.predict(data_test)
    
    for i in xrange(0, len(y_pred)):
        if pretype!=2:
            if y_pred[i]>1:
                Pre_scores[i][0]=1
            elif y_pred[i]<0:
                Pre_scores[i][0]=0
            else:
                Pre_scores[i][0] = y_pred[i]
        else:
            if y_pred[i]>100:
                Pre_scores[i][0]=100
            elif y_pred[i]<0:
                Pre_scores[i][0]=0
            else:
                Pre_scores[i][0] = y_pred[i]

    ###libsvm
    Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
    fea_apply = test_fea[:, Fea_index]

    #output_matrix(fea_apply,'C:/Users/penn/Desktop/aws/log_step1')
    if phmmtype == 1:
        Tr1 = pickle.load(open(model_path + 'trained_final_Tr1_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        Tr2 = pickle.load(open(model_path + 'trained_final_Tr2_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        Et1 = pickle.load(open(model_path + 'trained_final_Et1_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        Et2 = pickle.load(open(model_path + 'trained_final_Et2_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        lgps_te1 = pHMM_fea(Tr1, Et1, test_seqs[:, 0:20], group, 1, au, ad)
        lgps_te2 = pHMM_fea(Tr2, Et2, test_seqs[:, 20:39], group, 2, au, ad)

        fea_all = np.hstack((fea_apply, lgps_te1, lgps_te2))
    elif phmmtype == 0:
        fea_all = fea_apply
    
    score = test_scores
    file_path = libsvm_dataformat(fea_all, score, 'test', data_index, pretype, featype)
    Sco, Fea = svm_read_problem(file_path)
    m = svm_load_model(model_path + 
        'trained_final_libsvm_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(phmmtype) + '.model')
    y_test = Sco
    fea_test = Fea
    p_label_a, p_acc, p_val = svm_predict(y_test, fea_test, m)
    for i in xrange(0, len(p_label_a)):
        if pretype!=2:
            if p_label_a[i]>1:
                Pre_scores[i][1] = 1
            elif p_label_a[i]<0:
                Pre_scores[i][1] = 0
            else:
                Pre_scores[i][1] = p_label_a[i]
        else:
            if p_label_a[i]>100:
                Pre_scores[i][1] = 100
            elif p_label_a[i]<0:
                Pre_scores[i][1] = 0
            else:
                Pre_scores[i][1] = p_label_a[i]   
        Pre_scores[i][2] = k * Pre_scores[i][0] + (1 - k) * Pre_scores[i][1]

    return Pre_scores


def seq_reverse(seq):
    seq_ori = Seq(seq)
    seq_rev = seq_ori.reverse_complement()
    return seq_rev._data


def creat_seq_feature(spacer_ex):
    L = len(spacer_ex)
    seqs_20 = []
    seqs_30 = []
    for i in xrange(0, L):
        seq = spacer_ex[i]
        seqs_20.append(seq[4:24])
        seqs_30.append(seq)

    TM_region = TM_cal(seqs_30[0])
    for i in xrange(1, L):
        tm_region = TM_cal(seqs_30[i])
        TM_region = np.vstack((TM_region, tm_region))

    nc1, pos_fea1 = ncf(seqs_30[0], 1)

    for i in xrange(1, L):
        nc_f, pos_fea = ncf(seqs_30[i], 1)
        nc1 = np.vstack((nc1, nc_f))
        pos_fea1 = np.vstack((pos_fea1, pos_fea))

    nc2, pos_fea2 = ncf(seqs_30[0], 2)
    for i in xrange(1, L):
        nc_f, pos_fea = ncf(seqs_30[i], 2)
        nc2 = np.vstack((nc2, nc_f))
        pos_fea2 = np.vstack((pos_fea2, pos_fea))

    nc3, a = ncf(seqs_30[0], 3)
    for i in xrange(1, L):
        nc_f, a = ncf(seqs_30[i], 3)
        nc3 = np.vstack((nc3, nc_f))

    NC = np.hstack((nc1, nc2, nc3))
    Pos_fea = np.hstack((pos_fea1, pos_fea2))
    GC_fea = np.zeros((L, 2))
    for i in xrange(0, L):
        seq = seqs_30[i]
        GCcounts, GCcontent = countGC(seq, 0, 30)
        GC_fea[i][0] = GCcounts
        GC_fea[i][1] = GCcontent

    return TM_region, NC, Pos_fea, GC_fea


def find_all(a_string, sub):
    result = []
    k = 0
    while k < len(a_string):
        k = a_string.find(sub, k)
        if k == -1:
            return result
        else:
            result.append(k)
            k += 1  # change to k += len(sub) to not search overlapping results
    return result


contents = ['GRC', 'exon', 'intron', 'utr3', 'utr5', 'cds', 'cdna', 'peptide']


def obtain_fasta(filepath):
    fasta = {}
    with open(filepath) as file_one:
        for line in file_one:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta:
                    fasta[active_sequence_name] = []
                continue
            sequence = line
            fasta[active_sequence_name].append(sequence)

    return fasta


def extract_spacers_selfdef(fasta):
    fasta_content = {}
    keys = fasta.keys()
    spacer_all = []
    spacer_ex_all = []
    print(len(keys))
    # key = keys[0]
    for key in keys:
        seqs = fasta[key]
        seq = ''
        for j in xrange(0, len(seqs)):
            seq = seq + seqs[j]
        fasta_content[key]=seq

        genoSeq = str.upper(fasta_content[key])
        pos_sen = find_all(genoSeq, 'GG')
        pos_ant = find_all(genoSeq, 'CC')
        spacer = []
        spacer_ex = []
        for i in xrange(0, len(pos_sen)):
            start_p = pos_sen[i] - 21
            end_p = pos_sen[i] - 2
            start_p_ex = pos_sen[i] - 25
            end_p_ex = pos_sen[i] + 4
            if start_p_ex > 0 and end_p_ex < len(genoSeq):
                spacer.append(genoSeq[start_p:end_p + 1])
                spacer_ex.append(genoSeq[start_p_ex:end_p_ex + 1])

        spacer_anti = []
        spacer_anti_ex = []
        for i in xrange(0, len(pos_ant)):
            start_p = pos_ant[i] + 3
            end_p = pos_ant[i] + 22
            start_p_ex = pos_ant[i] - 3
            end_p_ex = pos_ant[i] + 26
            if start_p_ex > 0 and end_p_ex < len(genoSeq):
                s_an = Seq(genoSeq[start_p:end_p + 1]).reverse_complement()
                s_an_e = Seq(genoSeq[start_p_ex:end_p_ex + 1]).reverse_complement()
                spacer_anti.append(s_an._data)
                spacer_anti_ex.append(s_an_e._data)

        spacer_all += spacer
        spacer_all += spacer_anti
        spacer_ex_all += spacer_ex
        spacer_ex_all += spacer_anti_ex
        
    return spacer_all, spacer_ex_all


def extract_spacers_geno(fasta):
    fasta_content = {}
    keys = fasta.keys()
    key = keys[0]
    seqs = fasta[key]
    seq = ''
    for j in xrange(0, len(seqs)):
        seq = seq + seqs[j]
    fasta_content[key]=seq

    genoSeq = str.upper(fasta_content[key])
    pos_sen = find_all(genoSeq, 'GG')
    pos_ant = find_all(genoSeq, 'CC')
    spacer = []
    spacer_ex = []
    cut_fea = np.zeros((1, 6))
    for i in xrange(0, len(pos_sen)):
        start_p = pos_sen[i] - 21
        end_p = pos_sen[i] - 2
        start_p_ex = pos_sen[i] - 25
        end_p_ex = pos_sen[i] + 4
        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            cut_Fea = np.zeros((1, 6))
            spacer.append(genoSeq[start_p:end_p + 1])
            spacer_ex.append(genoSeq[start_p_ex:end_p_ex + 1])
            cut_geno = pos_sen[i] - 4
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            cut_fea = np.row_stack((cut_fea, cut_Fea))
    cut_fea_sen = cut_fea[1:len(cut_fea[:, 0]), :]

    spacer_anti = []
    spacer_anti_ex = []
    cut_fea = np.zeros((1, 6))
    for i in xrange(0, len(pos_ant)):
        start_p = pos_ant[i] + 3
        end_p = pos_ant[i] + 22
        start_p_ex = pos_ant[i] - 3
        end_p_ex = pos_ant[i] + 26
        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            cut_Fea = np.zeros((1, 6))
            s_an = Seq(genoSeq[start_p:end_p + 1]).reverse_complement()
            s_an_e = Seq(genoSeq[start_p_ex:end_p_ex + 1]).reverse_complement()
            spacer_anti.append(s_an._data)
            spacer_anti_ex.append(s_an_e._data)
            cut_geno = pos_ant[i] + 6
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            cut_fea = np.row_stack((cut_fea, cut_Fea))
    cut_fea_anti = cut_fea[1:len(cut_fea[:, 0]), :]

    spacer_all = spacer + spacer_anti
    spacer_ex_all = spacer_ex + spacer_anti_ex
    cut_fea_all = np.vstack((cut_fea_sen, cut_fea_anti))

    return spacer_all, spacer_ex_all, cut_fea_all


def extract_spacers_ensembl(fasta):
    fasta_content = {}
    keys = fasta.keys()
    for i in xrange(0, len(keys)):
        key = keys[i]
        for i in xrange(0, len(contents)):
            if key.find(contents[i]) > 0:
                newkey = contents[i]
                if newkey not in fasta_content:
                    fasta_content[newkey] = []
        seqs = fasta[key]
        seq = ''
        for j in xrange(0, len(seqs)):
            seq = seq + seqs[j]
        fasta_content[newkey].append(seq)

    genoSeq = str.upper(fasta_content['GRC'][0])
    exon_pos = []
    intron_pos = []
    if 'exon' in fasta_content:
        exons = fasta_content['exon']
        exon_pos = np.zeros((len(exons), 2))
        for i in xrange(0, len(exons)):
            exon = str.upper(exons[i])
            start = int(genoSeq.find(exon) + 1)  # seqence start from 1 not 0
            end = (start + len(exon))
            exon_pos[i][0] = start
            exon_pos[i][1] = end - 1
    if len(exon_pos) > 0:
        Exon_pos = sorted(exon_pos, key=lambda row: row[0])
        exon_pos = np.zeros((len(Exon_pos), 2))
        for i in xrange(0, len(Exon_pos)):
            exon_pos[i][0] = Exon_pos[i][0]
            exon_pos[i][1] = Exon_pos[i][1]

    if 'intron' in fasta_content:
        introns = fasta_content['intron']
        intron_pos = np.zeros((len(introns), 2))
        for i in xrange(0, len(introns)):
            intron = str.upper(introns[i])
            start = int(genoSeq.find(intron) + 1)  # seqence start from 1 not 0
            end = int(start + len(intron))
            intron_pos[i][0] = start
            intron_pos[i][1] = end - 1
    if len(intron_pos) > 0:
        Intron_pos = sorted(intron_pos, key=lambda row: row[0])
        intron_pos = np.zeros((len(Intron_pos), 2))
        for i in xrange(0, len(Intron_pos)):
            intron_pos[i][0] = Intron_pos[i][0]
            intron_pos[i][1] = Intron_pos[i][1]

    pos_sen = find_all(genoSeq, 'GG')
    pos_ant = find_all(genoSeq, 'CC')

    spacer = []
    spacer_ex = []
    cut_fea = np.zeros((1, 6))
    for i in xrange(0, len(pos_sen)):
        start_p = pos_sen[i] - 21
        end_p = pos_sen[i] - 2
        start_p_ex = pos_sen[i] - 25
        end_p_ex = pos_sen[i] + 4
        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            cut_Fea = np.zeros((1, 6))
            spacer.append(genoSeq[start_p:end_p + 1])
            spacer_ex.append(genoSeq[start_p_ex:end_p_ex + 1])
            cut_geno = pos_sen[i] - 4
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            flag = 0
            exon_num = 0
            if len(exon_pos) > 0:
                for j in xrange(0, len(exon_pos[:, 0])):
                    if cut_geno >= exon_pos[j, 0] and cut_geno <= exon_pos[j, 1]:  # cut at exon

                        flag = 1
                        exon_num = j
            if flag == 1:
                if exon_num == 0:
                    cut_trans = cut_geno
                else:
                    cut_trans = cut_geno
                    for j in xrange(0, exon_num):
                        cut_trans = cut_trans - (intron_pos[j, 1] - intron_pos[j, 0] + 1)
                cut_trans_per = 100 * ((cut_trans - 1) / len(fasta_content['cdna'][0]))
                cut_Fea[0][2] = cut_trans
                cut_Fea[0][3] = cut_trans_per
                len_utr5 = len(fasta_content['utr5'][0])
                len_protein = len(fasta_content['peptide'][0])
                cut_pro = math.ceil(((cut_trans - len_utr5 - 1) / 3))
                cut_pro_per = 100 * cut_pro / len_protein

                if cut_pro_per > 100 or cut_pro_per < 0:
                    cut_pro = 0
                    cut_pro_per = 0

                cut_Fea[0][4] = cut_pro
                cut_Fea[0][5] = cut_pro_per

            cut_fea = np.row_stack((cut_fea, cut_Fea))

    cut_fea_sen = cut_fea[1:len(cut_fea[:, 0]), :]

    spacer_anti = []
    spacer_anti_ex = []
    cut_fea = np.zeros((1, 6))
    for i in xrange(0, len(pos_ant)):
        start_p = pos_ant[i] + 3
        end_p = pos_ant[i] + 22
        start_p_ex = pos_ant[i] - 3
        end_p_ex = pos_ant[i] + 26
        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            cut_Fea = np.zeros((1, 6))
            s_an = Seq(genoSeq[start_p:end_p + 1]).reverse_complement()
            s_an_e = Seq(genoSeq[start_p_ex:end_p_ex + 1]).reverse_complement()
            spacer_anti.append(s_an._data)
            spacer_anti_ex.append(s_an_e._data)
            cut_geno = pos_ant[i] + 6
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            flag = 0
            exon_num = 0
            if len(exon_pos) > 0:
                for j in xrange(0, len(exon_pos[:, 0])):
                    if cut_geno >= exon_pos[j, 0] and cut_geno <= exon_pos[j, 1]:  # cut at exon
                        flag = 1
                        exon_num = j

            if flag == 1:
                if exon_num == 0:
                    cut_trans = cut_geno
                else:
                    cut_trans = cut_geno
                    for j in xrange(0, exon_num):
                        cut_trans = cut_trans - (intron_pos[j, 1] - intron_pos[j, 0] + 1)
                cut_trans_per = 100 * ((cut_trans - 1) / len(fasta_content['cdna'][0]))
                cut_Fea[0][2] = cut_trans
                cut_Fea[0][3] = cut_trans_per
                len_utr5 = len(fasta_content['utr5'][0])
                len_protein = len(fasta_content['peptide'][0])
                cut_pro = math.ceil(((cut_trans - len_utr5 - 1) / 3))
                cut_pro_per = 100 * cut_pro / len_protein

                if cut_pro_per > 100 or cut_pro_per < 0:
                    cut_pro = 0
                    cut_pro_per = 0

                cut_Fea[0][4] = cut_pro
                cut_Fea[0][5] = cut_pro_per

            cut_fea = np.row_stack((cut_fea, cut_Fea))

    cut_fea_anti = cut_fea[1:len(cut_fea[:, 0]), :]

    spacer_all = spacer + spacer_anti
    spacer_ex_all = spacer_ex + spacer_anti_ex
    cut_fea_all = np.vstack((cut_fea_sen, cut_fea_anti))

    return spacer_all, spacer_ex_all, cut_fea_all


def read_fasta(filepath, filetype):

    fasta = obtain_fasta(filepath)
    if filetype == 'annotated':
        spacer_all, spacer_ex_all, cut_fea_all = extract_spacers_ensembl(fasta)
    elif filetype == 'genome':
        spacer_all, spacer_ex_all, cut_fea_all = extract_spacers_geno(fasta)
    else:
        spacer_all, spacer_ex_all = extract_spacers_selfdef(fasta)
        cut_fea_all = []

    return spacer_all, spacer_ex_all, cut_fea_all


def independent_tests(train_seqs, train_fea, train_scores, test_seqs, test_fea, test_scores, data_index, pretype,
                      featype, phmmtype, splits):
    data_train = xgb.DMatrix(train_fea, train_scores)
    data_test = xgb.DMatrix(test_fea, test_scores)
    predict_scores = np.zeros((len(test_scores), 3))
    results = []
    k = 0.5
    if pretype == 1:
        param = {'lambda': 800, 'objective': 'reg:tweedie', 'nthread': 8, 'booster': 'gbtree', 'silent': 1}
        bst = xgb.train(param, data_train, num_boost_round=450)
        filename = model_path + 'trained_final_model_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(
            phmmtype) + '.model'
        bst.save_model(filename)
        y_pred = bst.predict(data_test)
        # Y_pred=np.zeros((len(y_pred),1))
        for i in range(0, len(y_pred)):
            predict_scores[i][0] = y_pred[i]

        Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(data_index) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        fea_apply_tr = train_fea[:, Fea_index]
        fea_apply_te = test_fea[:, Fea_index]

        if phmmtype == 1:
            lgps_tr1, lgps_te1 = pHMM_fea_all(train_seqs[:,0:20], train_scores, test_seqs[:,0:20], 1, 20, 1, 20)
            lgps_tr2, lgps_te2 = pHMM_fea_all(train_seqs[:,20:39], train_scores, test_seqs[:,20:39], 2, 20, 1, 20)

            fea_all_tr = np.hstack((fea_apply_tr, lgps_tr1, lgps_tr2))
            fea_all_te = np.hstack((fea_apply_te, lgps_te1, lgps_te2))
        elif phmmtype == 0:
            fea_all_tr = fea_apply_tr
            fea_all_te = fea_apply_te

        score_tr = train_scores
        score_te = test_scores
        file_path_tr = libsvm_dataformat(fea_all_tr, score_tr, 'train', data_index, pretype, featype)
        file_path_te = libsvm_dataformat(fea_all_te, score_te, 'test', data_index, pretype, featype)
        Sco_tr, Fea_tr = svm_read_problem(file_path_tr)
        Sco_te, Fea_te = svm_read_problem(file_path_te)
        C = -3
        G = -3
        para = '-s 3 -t 2 -p ' + str(0.14) + ' -c ' + str(2 ** C) + ' -g ' + str(2 ** G)
        m = svm_train(Sco_tr, Fea_tr, para)
        svm_save_model(model_path + 'trained_libsvm' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(
            phmmtype) + '.model', m)
        p_label_a, p_acc, p_val = svm_predict(Sco_te, Fea_te, m)
        for i in xrange(0, len(p_label_a)):
            predict_scores[i][1] = p_label_a[i]
            predict_scores[i][2] = k * predict_scores[i][0] + (1 - k) * predict_scores[i][1]

        splitnum = len(splits[0, :])
        Sprs = []
        spr_all = []
        for i in xrange(0, splitnum):
            spr = np.zeros((1, 3))
            index = np.where(splits[:, i] == 0)
            spr1 = stats.spearmanr(test_scores[index[0], :], predict_scores[index[0], 0])
            spr2 = stats.spearmanr(test_scores[index[0], :], predict_scores[index[0], 1])
            spr3 = stats.spearmanr(test_scores[index[0], :], predict_scores[index[0], 2])
            spr[0, 0] = spr1[0]
            spr[0, 1] = spr2[0]
            spr[0, 2] = spr3[0]
            print spr
            Sprs.append(spr)
        spr_all.append(stats.spearmanr(test_scores, predict_scores[:, 0]))
        spr_all.append(stats.spearmanr(test_scores, predict_scores[:, 1]))
        spr_all.append(stats.spearmanr(test_scores, predict_scores[:, 2]))
        print spr_all
        results.append(spr)
        results.append(spr_all)

    elif pretype == 3:
        param = {'max_depth': 4, 'eta': 0.12, 'silent': 1, 'objective': 'binary:logistic', 'booster': 'gbtree'}
        bst = xgb.train(param, data_train, num_boost_round=450)
        filename = model_path + 'trained_final_model_' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(
            phmmtype) + '.model'
        bst.save_model(filename)
        y_pred = bst.predict(data_test)
        # Y_pred=np.zeros((len(y_pred),1))
        for i in range(0, len(y_pred)):
            predict_scores[i][0] = y_pred[i]

        Fea_index = pickle.load(open(model_path + 'Fea_index_' + str(5) + '_' + str(pretype) + '_' + str(featype), 'rb'))
        fea_apply_tr = train_fea[:, Fea_index]
        fea_apply_te = test_fea[:, Fea_index]

        if phmmtype == 1:
            TRs, ETs, phmm_tr = pHmmGE(train_seqs, train_scores, train_seqs, 1, 20)
            phmm_te = phmm_scores(test_seqs, TRs, ETs)

            fea_all_tr = np.hstack((phmm_tr, fea_apply_tr))
            fea_all_te = np.hstack((phmm_te, fea_apply_te))
        elif phmmtype == 0:
            fea_all_tr = fea_apply_tr
            fea_all_te = fea_apply_te

        score_tr = train_scores
        score_te = test_scores
        file_path_tr = libsvm_dataformat(fea_all_tr, score_tr, 'train', data_index, pretype, featype)
        file_path_te = libsvm_dataformat(fea_all_te, score_te, 'test', data_index, pretype, featype)
        Sco_tr, Fea_tr = svm_read_problem(file_path_tr)
        Sco_te, Fea_te = svm_read_problem(file_path_te)
        C = 1
        G = -5
        para = '-t 2 -b 1 -p 0.14' + ' -c ' + str(2 ** C) + ' -g ' + str(2 ** G)
        m = svm_train(Sco_tr, Fea_tr, para)
        svm_save_model(model_path + 'trained_libsvm' + str(data_index) + '_' + str(pretype) + '_' + str(featype) + '_' + str(
            phmmtype) + '.model', m)
        p_label_a, p_acc, p_val = svm_predict(Sco_te, Fea_te, m, '-b 1')
        for i in xrange(0, len(p_label_a)):
            predict_scores[i][1] = p_val[i][0]
            predict_scores[i][2] = k * predict_scores[i][0] + (1 - k) * predict_scores[i][1]

        result1 = evaluate_performance(predict_scores[:, 0], test_scores, 1, 0, 0.5)
        result2 = evaluate_performance(predict_scores[:, 1], test_scores, 1, 0, 0.5)
        result3 = evaluate_performance(predict_scores[:, 2], test_scores, 1, 0, 0.5)
        results=np.hstack((result1, result2, result3))
    return results


def train_final_models(data_index, pretype, featype, phmmtype):
    train_seq_20 = []
    train_seq_30 = []
    if pretype == 3:
        data_index = 8
        matfn = data_path + 'classification_data_sets_new.mat'  # the path of .mat data
        data = sio.loadmat(matfn)

        seq_xu_rb_p = data['xu_seq_rb_p']
        seq_xu_rb_n = data['xu_seq_rb_n']
        xu_cut_fea_rb_p = data['xu_cut_fea_rb_p']
        xu_cut_fea_rb_n = data['xu_cut_fea_rb_n']

        seq_xu_nrb_p = data['xu_seq_nrb_p']
        seq_xu_nrb_n = data['xu_seq_nrb_n']
        xu_cut_fea_nrb_p = data['xu_cut_fea_nrb_p']
        xu_cut_fea_nrb_n = data['xu_cut_fea_nrb_n']

        seq_xu_mE_p = data['xu_seq_mE_p']
        seq_xu_mE_n = data['xu_seq_mE_n']
        xu_cut_fea_mE_p = data['xu_cut_fea_mE_p']
        xu_cut_fea_mE_n = data['xu_cut_fea_mE_n']

        train_seq_p = np.vstack((seq_xu_rb_p, seq_xu_nrb_p, seq_xu_mE_p))
        train_seq_n = np.vstack((seq_xu_rb_n, seq_xu_nrb_n, seq_xu_mE_n))

        seq = np.vstack((train_seq_p, train_seq_n))
        cut_fea = np.vstack(
            (xu_cut_fea_rb_p, xu_cut_fea_nrb_p, xu_cut_fea_mE_p, xu_cut_fea_rb_n, xu_cut_fea_nrb_n, xu_cut_fea_mE_n))
        labels = np.zeros((len(seq), 1))
        labels[0:len(train_seq_p), 0] = 1
    else:

        matfn = data_path + 'regression_data_sets_new.mat'  # the path of .mat data
        data = sio.loadmat(matfn)
        if pretype == 1:  # human and mouse
            data_index = 3
            seq = data['D_Integrate_4379_new'].tolist()
            cut_fea = data['cut_fea_4379']
            score = data['train_D_Integrate_4379_score']


        elif pretype == 2:  # zebrafish
            data_index = 4
            seq = data['gScan_1020'].tolist()
            cut_fea = data['cut_fea_gScan']
            score = data['gScan_score']

    for i in xrange(0, len(seq)):
        train_seq_20.append(seq[i][0][0].encode('utf-8'))
        train_seq_30.append(seq[i][1][0].encode('utf-8'))

    TM_region_tr, NC_tr, Pos_fea_tr, GC_fea_tr = creat_seq_feature(train_seq_30)
    train_fea = []

    Train_fea = np.hstack((TM_region_tr, NC_tr, GC_fea_tr, Pos_fea_tr, cut_fea))
        
    if featype == 1:  # with all cut fea
        train_fea = Train_fea
    elif featype == 2:  # without cut fea
        train_fea = Train_fea[:, 0:674]
    elif featype == 3:  # with cut_geno
        train_fea = Train_fea[:, 0:675]

    alpStat_tr1 = binary_seqs(train_seq_20, 1)
    alpStat_tr2 = binary_seqs(train_seq_20, 2)
    alpStat_tr = np.hstack((alpStat_tr1, alpStat_tr2))

    if pretype == 1:
        data_index, pretype, featype, phmmtype=train_model_regression(alpStat_tr, train_fea, score, 3, pretype, featype, phmmtype, 800, 6, 450, 0.14,
                               -3, -3, 20, 1, 20)
    elif pretype == 2:
        data_index, pretype, featype, phmmtype=train_model_regression(alpStat_tr, train_fea, score, 4, pretype, featype, phmmtype, 700, 6, 100, 0.14,
                               8, -6, 20, 1, 20)
    elif pretype == 3:
        data_index, pretype, featype, phmmtype=train_model_classification(alpStat_tr, train_fea, labels, 8, pretype, featype, phmmtype, 0.12, 4, 450,
                                   0.14, 1, -5, 1, 20)

    return data_index, pretype, featype, phmmtype


def cross_validations(data_index, pretype, featype):
    if pretype < 3:
        matfn = data_path + 'regression_data_sets_new.mat'
        data = sio.loadmat(matfn)
        if pretype == 1:  # human and mouse
            if data_index == 1:
                seq = data['D_2014_1830_new'].tolist()
                cut_fea = data['cut_fea_1830']
                score = data['train_D_2014_1830_score']
                splits = data['split_1830']
            elif data_index == 2:
                seq = data['D_2016_2549_new'].tolist()
                cut_fea = data['cut_fea_2549']
                score = data['train_D_2016_2549_score']
                splits = data['split_2549']
            elif data_index == 3:
                seq = data['D_Integrate_4379_new'].tolist()
                cut_fea = data['cut_fea_4379']
                score = data['train_D_Integrate_4379_score']
                splits = data['split_4379']
        elif pretype == 2:  # zebrafish
            data_index == 4
            seq = data['gScan_1020'].tolist()
            cut_fea = data['cut_fea_gScan']
            score = data['gScan_score']
            splits = data['split_gScan_10']
    elif pretype == 3:
        matfn = data_path + 'classification_data_sets_new.mat'  # the path of .mat data
        data = sio.loadmat(matfn)
        if data_index == 5:
            seq_xu_rb_p = data['xu_seq_rb_p'].tolist()
            seq_xu_rb_n = data['xu_seq_rb_n'].tolist()
            xu_cut_fea_rb_p = data['xu_cut_fea_rb_p']
            xu_cut_fea_rb_n = data['xu_cut_fea_rb_n']
            splits = data['xu_split']
            seq = np.vstack((seq_xu_rb_p, seq_xu_rb_n))
            cut_fea = np.vstack((xu_cut_fea_rb_p, xu_cut_fea_rb_n))
            labels = np.zeros((len(splits[:, 0]), 1))
            labels[0:len(seq_xu_rb_p), 0] = 1
        elif data_index == 10:
            seq_chari_sp_p = data['chari_seq_sp_p'].tolist()
            seq_chari_sp_n = data['chari_seq_sp_n'].tolist()
            chari_cut_fea_sp_p = data['chari_cut_fea_sp_p']
            chari_cut_fea_sp_n = data['chari_cut_fea_sp_n']
            splits = data['chari_sp_split']
            seq = np.vstack((seq_chari_sp_p, seq_chari_sp_n))
            cut_fea = np.vstack((chari_cut_fea_sp_p, chari_cut_fea_sp_n))
            labels = np.zeros((len(splits[:, 0]), 1))
            labels[0:len(seq_chari_sp_p), 0] = 1

        elif data_index == 11:
            featype = 2
            seq_chari_st_p = data['chari_seq_st_p'].tolist()
            seq_chari_st_n = data['chari_seq_st_n'].tolist()
            splits = data['chari_st_split']
            cut_fea = np.zeros((len(splits[:, 0]), 3))
            seq = np.vstack((seq_chari_st_p, seq_chari_st_n))
            labels = np.zeros((len(splits[:, 0]), 1))
            labels[0:len(seq_chari_st_p), 0] = 1

    train_seq_20 = []
    train_seq_30 = []
    for i in xrange(0, len(seq)):
        train_seq_20.append(seq[i][0][0].encode('utf-8'))
        train_seq_30.append(seq[i][1][0].encode('utf-8'))
    if data_index != 11:  # the extend sequence of chari_st is 27bp in length
        TM_region_tr, NC_tr, Pos_fea_tr, GC_fea_tr = creat_seq_feature(train_seq_30)
        Train_fea = np.hstack((TM_region_tr, NC_tr, GC_fea_tr, Pos_fea_tr, cut_fea))
        if featype == 1:  # with all cut fea
            train_fea = Train_fea
        elif featype == 2:  # without cut fea
            train_fea = Train_fea[:, 0:674]
        elif featype == 3:  # with cut_geno
            train_fea = Train_fea[:, 0:675]
    else:
        TM_region_tr = np.zeros((len(train_seq_30), 4))
        for i in range(0, len(train_seq_30)):
            s_20 = train_seq_20[i]
            s_all = train_seq_30[i]
            tm_sub1 = mt.Tm_NN(Seq(s_20[2:7]))  # TMr7
            tm_sub2 = mt.Tm_NN(Seq(s_20[7:15]))  # TMr6
            tm_sub3 = mt.Tm_NN(Seq(s_20[15:20]))  # TMr5
            tm_sub4 = mt.Tm_NN(Seq(s_all))  # TMr4

            TM_region_tr[i][0] = tm_sub1
            TM_region_tr[i][1] = tm_sub2
            TM_region_tr[i][2] = tm_sub3
            TM_region_tr[i][3] = tm_sub4

        nc1_tr, pos_fea1_tr = ncf(train_seq_30[0], 1)

        for i in range(1, len(train_seq_30)):
            nc_f, pos_fea = ncf(train_seq_30[i], 1)
            nc1_tr = np.vstack((nc1_tr, nc_f))
            pos_fea1_tr = np.vstack((pos_fea1_tr, pos_fea))

        nc2_tr, pos_fea2_tr = ncf(train_seq_30[0], 2)
        for i in range(1, len(train_seq_30)):
            nc_f, pos_fea = ncf(train_seq_30[i], 2)
            nc2_tr = np.vstack((nc2_tr, nc_f))
            pos_fea2_tr = np.vstack((pos_fea2_tr, pos_fea))

        nc3_tr, a = ncf(train_seq_30[0], 3)
        for i in range(1, len(train_seq_30)):
            nc_f, a = ncf(train_seq_30[i], 3)
            nc3_tr = np.vstack((nc3_tr, nc_f))

        GC_fea_tr = np.zeros((len(train_seq_30), 2))
        for i in range(0, len(train_seq_30)):
            seq = train_seq_30[i]
            GCcounts, GCcontent = countGC(seq, 0, len(train_seq_30[0]))
            GC_fea_tr[i][0] = GCcounts
            GC_fea_tr[i][1] = GCcontent
        NC_tr = np.hstack((nc1_tr, nc2_tr, nc3_tr))
        Pos_fea_tr = np.hstack((pos_fea1_tr, pos_fea2_tr))
        train_fea = np.hstack((TM_region_tr, NC_tr, Pos_fea_tr, GC_fea_tr))

    alpStat_tr1 = binary_seqs(train_seq_20, 1)
    alpStat_tr2 = binary_seqs(train_seq_20, 2)
    alpStat_tr = np.hstack((alpStat_tr1, alpStat_tr2))

    if pretype == 1:
        Importance, mean_auc, pre_score_xg = cross_validation_xg(train_fea, score, splits, 800, 450, 6, 1)
        Mean_AUC, Pre_score_svm, Fea_index = cross_validation_libsvm(alpStat_tr, Importance, train_fea, score, splits,
                                                                     -3, -3, 0.14, 70, 1, 20, 1, 20)
        Sprs, Pre_score_ens = cross_validation_ens(splits, score, pre_score_xg, Pre_score_svm, 1, 0.5)
        print Sprs
    elif pretype == 2:
        Importance, mean_auc, pre_score_xg = cross_validation_xg(train_fea, score, splits, 700, 100, 6, 2)
        Mean_AUC, Pre_score_svm, Fea_index = cross_validation_libsvm(alpStat_tr, Importance, train_fea, score, splits,
                                                                     8, -6, 0.14, 30, 2, 20, 1, 20)
        Sprs, Pre_score_ens = cross_validation_ens(splits, score, pre_score_xg, Pre_score_svm, 2, 0.5)
        print Sprs
    elif pretype == 3:
        if data_index == 5:
            Importance, mean_auc, pre_score_xg = cross_validation_cls_xg(train_fea, labels, splits, 0.12, 450, 4)
            Mean_AUC, Pre_score_svm, Fea_index = cross_validation_cls_libsvm(alpStat_tr, Importance, train_fea, labels,
                                                                             splits, 1, -5, 50, 1, 20)
            AUC, Pre_score_ens, Pre_Scores = cross_validation_cls_ens(splits, labels, pre_score_xg, Pre_score_svm, 0.5)
            print AUC
        elif data_index == 10:
            Importance, mean_auc, pre_score_xg = cross_validation_cls_xg(train_fea, labels, splits, 0.12, 100, 4)
            Mean_AUC, Pre_score_svm, Fea_index = cross_validation_cls_libsvm(alpStat_tr, Importance, train_fea, labels,
                                                                             splits, 2, -5, 20, 1, 20)
            AUC, Pre_score_ens, Pre_Scores = cross_validation_cls_ens(splits, labels, pre_score_xg, Pre_score_svm, 0.5)
            print AUC
        elif data_index == 11:
            Importance, mean_auc, pre_score_xg = cross_validation_cls_xg(train_fea, labels, splits, 0.1, 200, 8)
            Mean_AUC, Pre_score_svm, Fea_index = cross_validation_cls_libsvm(alpStat_tr, Importance, train_fea, labels,
                                                                             splits, 1, -2, 20, 1, 20)
            AUC, Pre_score_ens, Pre_Scores = cross_validation_cls_ens(splits, labels, pre_score_xg, Pre_score_svm, 0.5)
            print AUC

    return data_index, pretype, featype


def test_final_models(data_index, pretype, featype, phmmtype):
    if pretype == 1:
        matfn = data_path + 'regression_data_sets_new.mat'
        data = sio.loadmat(matfn)
        if data_index == 1:
            seq_train = data['D_2014_1830_new'].tolist()
            cut_fea_train = data['cut_fea_1830']
            score_train = data['train_D_2014_1830_score']

            seq_test = data['D_2016_2549_new'].tolist()
            cut_fea_test = data['cut_fea_2549']
            score_test = data['train_D_2016_2549_score']
            splits_test = data['split_2549']
        elif data_index == 2:
            seq_train = data['D_2016_2549_new'].tolist()
            cut_fea_train = data['cut_fea_2549']
            score_train = data['train_D_2016_2549_score']

            seq_test = data['D_2014_1830_new'].tolist()
            cut_fea_test = data['cut_fea_1830']
            score_test = data['train_D_2014_1830_score']
            splits_test = data['split_1830']
        elif data_index == 3:
            seq_train=data['D_Integrate_4379_new']
            cut_fea_train=data['cut_fea_4379']
            score_train=data['train_D_Integrate_4379_score']

            seq_test = data['case_seq']
            cut_fea_test = data['case_cut']
            score_test=data['case_score']
            splits_test=np.zeros((len(score_test),1))

        train_seq_20 = []
        train_seq_30 = []
        for i in xrange(0, len(seq_train)):
            train_seq_20.append(seq_train[i][0][0].encode('utf-8'))
            train_seq_30.append(seq_train[i][1][0].encode('utf-8'))

        test_seq_20 = []
        test_seq_30 = []
        for i in xrange(0, len(seq_test)):
            test_seq_20.append(seq_test[i][0][0].encode('utf-8'))
            test_seq_30.append(seq_test[i][1][0].encode('utf-8'))

        TM_region_tr, NC_tr, Pos_fea_tr, GC_fea_tr = creat_seq_feature(train_seq_30)
        TM_region_te, NC_te, Pos_fea_te, GC_fea_te = creat_seq_feature(test_seq_30)
        feaAll_train = np.hstack((TM_region_tr, NC_tr, GC_fea_tr, Pos_fea_tr, cut_fea_train))
        feaAll_test = np.hstack((TM_region_te, NC_te, GC_fea_te, Pos_fea_te, cut_fea_test))
        if featype == 1:
            train_fea = feaAll_train
            test_fea = feaAll_test
        elif featype == 2:
            train_fea = feaAll_train[:, 0:674]
            test_fea = feaAll_test[:, 0:674]
        elif featype == 3:
            train_fea = feaAll_train[:, 0:675]
            test_fea = feaAll_test[:, 0:675]

        alpStat_tr1 = binary_seqs(train_seq_20, 1)
        alpStat_tr2 = binary_seqs(train_seq_20, 2)
        alpStat_tr = np.hstack((alpStat_tr1, alpStat_tr2))
        alpStat_te1 = binary_seqs(test_seq_20, 1)
        alpStat_te2 = binary_seqs(test_seq_20, 2)
        alpStat_te = np.hstack((alpStat_te1, alpStat_te2))
        results = independent_tests(alpStat_tr, train_fea, score_train, alpStat_te, test_fea, score_test, data_index,
                                    pretype, featype, phmmtype, splits_test)
    elif pretype == 3:
        matfn = data_path + 'classification_data_sets_new.mat'  # the path of .mat data
        data = sio.loadmat(matfn)
        seq_xu_rb_p = data['xu_seq_rb_p'].tolist()
        seq_xu_rb_n = data['xu_seq_rb_n'].tolist()
        xu_cut_fea_rb_p = data['xu_cut_fea_rb_p']
        xu_cut_fea_rb_n = data['xu_cut_fea_rb_n']

        seq_xu_nrb_p = data['xu_seq_nrb_p'].tolist()
        seq_xu_nrb_n = data['xu_seq_nrb_n'].tolist()
        xu_cut_fea_nrb_p = data['xu_cut_fea_nrb_p']
        xu_cut_fea_nrb_n = data['xu_cut_fea_nrb_n']

        seq_xu_mE_p = data['xu_seq_mE_p'].tolist()
        seq_xu_mE_n = data['xu_seq_mE_n'].tolist()
        xu_cut_fea_mE_p = data['xu_cut_fea_mE_p']
        xu_cut_fea_mE_n = data['xu_cut_fea_mE_n']

        seq_xu_inde1_p = data['xu_seq_inde1_p'].tolist()
        seq_xu_inde1_n = data['xu_seq_inde1_n'].tolist()
        xu_cut_fea_inde1_p = data['xu_cut_fea_inde1_p']
        xu_cut_fea_inde1_n = data['xu_cut_fea_inde1_n']

        seq_xu_inde2_p = data['xu_seq_inde2_p'].tolist()
        seq_xu_inde2_n = data['xu_seq_inde2_n'].tolist()
        xu_cut_fea_inde2_p = data['xu_cut_fea_inde2_p']
        xu_cut_fea_inde2_n = data['xu_cut_fea_inde2_n']

        if data_index == 6:
            train_Seqs = np.vstack((seq_xu_rb_p, seq_xu_rb_n))
            train_cut_fea = np.vstack((xu_cut_fea_rb_p, xu_cut_fea_rb_n))
            train_labels = np.zeros((len(train_cut_fea[:, 0]), 1))
            train_labels[0:len(seq_xu_rb_p), :] = 1
            test_Seqs = np.vstack((seq_xu_nrb_p, seq_xu_nrb_n))
            test_cut_fea = np.vstack((xu_cut_fea_nrb_p, xu_cut_fea_nrb_n))
            test_labels = np.zeros((len(test_cut_fea[:, 0]), 1))
            test_labels[0:len(xu_cut_fea_nrb_p), :] = 1
        elif data_index == 7:
            train_Seqs = np.vstack((seq_xu_rb_p, seq_xu_nrb_p, seq_xu_rb_n, seq_xu_nrb_n))
            test_Seqs = np.vstack((seq_xu_mE_p, seq_xu_mE_n))
            train_cut_fea = np.vstack((xu_cut_fea_rb_p, xu_cut_fea_nrb_p, xu_cut_fea_rb_n, xu_cut_fea_nrb_n))
            train_labels = np.zeros((len(train_cut_fea[:, 0]), 1))
            train_labels[0:len(seq_xu_rb_p) + len(xu_cut_fea_nrb_p), :] = 1
            test_cut_fea = np.vstack((xu_cut_fea_mE_p, xu_cut_fea_mE_n))
            test_labels = np.zeros((len(test_cut_fea[:, 0]), 1))
            test_labels[0:len(xu_cut_fea_mE_p), :] = 1
        elif data_index == 8:
            train_Seqs = np.vstack((seq_xu_rb_p, seq_xu_nrb_p, seq_xu_mE_p, seq_xu_rb_n, seq_xu_nrb_n, seq_xu_mE_n))
            test_Seqs = np.vstack((seq_xu_inde1_p, seq_xu_inde1_n))
            train_cut_fea = np.vstack((xu_cut_fea_rb_p, xu_cut_fea_nrb_p, xu_cut_fea_mE_p, xu_cut_fea_rb_n,
                                       xu_cut_fea_nrb_n, xu_cut_fea_mE_n))
            train_labels = np.zeros((len(train_cut_fea[:, 0]), 1))
            train_labels[0:len(seq_xu_rb_p) + len(xu_cut_fea_nrb_p) + len(xu_cut_fea_mE_p), :] = 1
            test_cut_fea = np.vstack((xu_cut_fea_inde1_p, xu_cut_fea_inde1_n))
            test_labels = np.zeros((len(test_cut_fea[:, 0]), 1))
            test_labels[0:len(xu_cut_fea_inde1_p), :] = 1
        elif data_index == 9:
            train_Seqs = np.vstack((seq_xu_rb_p, seq_xu_nrb_p, seq_xu_mE_p, seq_xu_rb_n, seq_xu_nrb_n, seq_xu_mE_n))
            test_Seqs = np.vstack((seq_xu_inde2_p, seq_xu_inde2_n))
            train_cut_fea = np.vstack((xu_cut_fea_rb_p, xu_cut_fea_nrb_p, xu_cut_fea_mE_p, xu_cut_fea_rb_n,
                                       xu_cut_fea_nrb_n, xu_cut_fea_mE_n))
            train_labels = np.zeros((len(train_cut_fea[:, 0]), 1))
            train_labels[0:len(seq_xu_rb_p) + len(xu_cut_fea_nrb_p) + len(xu_cut_fea_mE_p), :] = 1
            test_cut_fea = np.vstack((xu_cut_fea_inde2_p, xu_cut_fea_inde2_n))
            test_labels = np.zeros((len(test_cut_fea[:, 0]), 1))
            test_labels[0:len(xu_cut_fea_inde2_p), :] = 1

        train_seq_20 = []
        train_seq_30 = []
        for i in xrange(0, len(train_Seqs)):
            seq_30=train_Seqs[i][1][0].encode('utf-8')
            train_seq_20.append(seq_30[4:24])
            train_seq_30.append(seq_30)

        test_seq_20 = []
        test_seq_30 = []
        for i in xrange(0, len(test_Seqs)):
            seq_30=test_Seqs[i][1][0].encode('utf-8')
            test_seq_20.append(seq_30[4:24])
            test_seq_30.append(seq_30)

        TM_region_tr, NC_tr, Pos_fea_tr, GC_fea_tr = creat_seq_feature(train_seq_30)
        TM_region_te, NC_te, Pos_fea_te, GC_fea_te = creat_seq_feature(test_seq_30)
        feaAll_train = np.hstack((TM_region_tr, NC_tr, GC_fea_tr, Pos_fea_tr, train_cut_fea))
        feaAll_test = np.hstack((TM_region_te, NC_te, GC_fea_te, Pos_fea_te, test_cut_fea))
        if featype == 1:
            train_fea = feaAll_train
            test_fea = feaAll_test
        elif featype == 2:
            train_fea = feaAll_train[:, 0:674]
            test_fea = feaAll_test[:, 0:674]
        elif featype == 3:
            train_fea = feaAll_train[:, 0:675]
            test_fea = feaAll_test[:, 0:675]

        alpStat_tr1 = binary_seqs(train_seq_20, 1)
        alpStat_tr2 = binary_seqs(train_seq_20, 2)
        alpStat_tr = np.hstack((alpStat_tr1, alpStat_tr2))
        alpStat_te1 = binary_seqs(test_seq_20, 1)
        alpStat_te2 = binary_seqs(test_seq_20, 2)
        alpStat_te = np.hstack((alpStat_te1, alpStat_te2))
        results = independent_tests(alpStat_tr, train_fea, train_labels, alpStat_te, test_fea, test_labels, data_index,
                                    pretype, featype, phmmtype, [])
    return results


def prediction_ce(filepath, filetype, species, pretype, sgtype, phmmtype):

    if species==2:
        pretype = 2
    
    spacer_all, spacer_ex_all, Cut_fea_all = read_fasta(filepath, filetype)
    if filetype == 'annotated':  # with all gene information
        featype = 1
    elif filetype == 'genome':  # with no protein information
        featype = 3
        sgtype = 0
    else:  # self defined sequence
        featype = 2
        sgtype = 0
        
    spacers_20 = []
    spacers_30 = []
    if sgtype == 1:  # with all gene information
# predict sgRNAs cutting at exon only
        index = np.where(Cut_fea_all[:, 5] > 0)
        if len(index[0])==0:
            index = np.where(Cut_fea_all[:, 5] == 0)
            
        for i in index[0]:
            spacers_20.append(spacer_all[i])
            spacers_30.append(spacer_ex_all[i])
        cut_feas = Cut_fea_all[index[0], :]
    else:
        spacers_20 = spacer_all
        spacers_30 = spacer_ex_all
        cut_feas = Cut_fea_all

    TM_region_te, NC_te, Pos_fea_te, GC_fea_te = creat_seq_feature(spacers_30)
    alpStat_te1 = binary_seqs(spacers_20, 1)
    alpStat_te2 = binary_seqs(spacers_20, 2)
    alpStat_te = np.hstack((alpStat_te1, alpStat_te2))
    scores = np.zeros((len(spacers_20), 1))
    if featype!=2:
        cut_fea_col = [1, 4, 5]
        cut_fea = cut_feas[:, cut_fea_col]
        Test_fea = np.hstack((TM_region_te, NC_te, GC_fea_te, Pos_fea_te, cut_fea))
    else:
        cut_fea=[]
        Test_fea = np.hstack((TM_region_te, NC_te, GC_fea_te, Pos_fea_te))

    if featype == 1:
        test_fea = Test_fea
    elif featype == 2:
        test_fea = Test_fea[:, 0:674]
    elif featype == 3:
        test_fea = Test_fea[:, 0:675]
    
    if pretype < 3:
        if pretype == 1:
            data_index = 3
        elif pretype == 2:
            data_index = 4
        Pre_scores = test_models_regression(test_fea, alpStat_te, scores, data_index, pretype, featype, phmmtype, 20,
                                            1, 20, 0.5)
    elif pretype == 3:
        data_index = 8
        Pre_scores = test_models_classification(test_fea, alpStat_te, scores, data_index, pretype, featype, phmmtype, 1,
                                                20, 0.5)
    
    score_index=np.zeros((len(Pre_scores[:,0]),1))
    for i in xrange(0, len(score_index)):
        score_index[i,0]=i
    Out=np.hstack((Pre_scores, score_index))
    out = sorted(Out, key=lambda row: row[2], reverse=True)
    file_path = 'predict_results/predict_results.csv'
    with open(file_path, 'w') as csv_file:
        fieldnames = ['spacer', 'xgboost_predict', 'libsvm_predict', 'TSAM_predict']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in xrange(0, len(spacers_20)):
            ind = int(out[i][3])                    
            writer.writerow(
                {'spacer': spacers_20[ind], 'xgboost_predict': Pre_scores[ind, 0], 'libsvm_predict': Pre_scores[ind, 1],
                 'TSAM_predict': Pre_scores[ind, 2]})

    return file_path, Pre_scores

def output_log(fp,log):
    with open(fp, 'w+') as file_out:
        file_out.write(log+'\n')

if __name__ == '__main__':
    # Five global parameters:
    #    data_index--11 training datasets together with their precomputed features and the cross-validation splits
    #           1----FC(doench 1830, logocv)
    #           2----RES(doench 2549, logocv)
    #           3----FC+RES(doench 4379, logocv)
    #           4----crisprScan(moreno 1020, logocv)
    #           5----xu_ribo(731H,438L, 3-fold cv)
    #           6----xu_non-ribo(671H,237L, intergeneset cv)
    #           7----xu_mouse(830H,234L, interplatform cv)
    #           8----xu_inde1(52H,25L, independent test)
    #           9----xu_inde2(110H,110L, independent test)
    #           10----Chari_spCas9(133H,146L, 10-fold cv)
    #           11----Chari_stlCas9(82H,69L, 10-fold cv)
    #    pretype--three types of predictions
    #           1---regression with the FC+RES trained model (default)
    #           2---regression with the crisprScan trained model
    #           3---classification with the xu_ribo+xu_non-ribo+xu_mouse trained model
    #    featype--Four Mutation types of our trained features
    #           1---training with all the features(677-d) (default)
    #           2---training with all but no cut position features(674-d)
    #           3---training with all but no cut position related protein features(675-d)
    #    phmmtype--whether apply the phmm feature(to speed up the prediction, off can be used)
    #           1--use (default)
    #           0--unuse
    #    sgtype---the output sgRNA types
    #           1---cut at exons only (default)
    #           0---all potential

    ##############################  example of cross-validation experiments ##################################################
    ## when test cross-validation, data_index can be 1, 2, 3, 4, 5, 10, 11
    ## when data_index=1, 2, 3,then, pretype =1
    ## when data_index=4, then, pretype=2
    ## when data_index=10, 11, then, pretype=3
    ## when data_index=11, then, featype =2
    ###  example code:

#    data_index = 4
#    pretype = 2
#    featype = 2
#    Data_index, Pretype, Featype = cross_validations(data_index, pretype, featype)
    ##########################################################################################################################

    ############################  examle of independent test on different models  ############################################
    ## when test independent tests, data_index can be 1, 2, 6, 7, 8, 9
    ## when data_index=1, 2 then pretype=1
    ## when data_index=6, 7, 8, 9 then pretype=3
    ###  example code
#     data_index = 3
#     pretype = 1
#     featype = 2
#     phmmtype = 1
#     results = test_final_models(data_index, pretype, featype, phmmtype)
#     print results
    ###########################################################################################################################

############################### codes for training the final model #############################################################
### three types of final models can be constructed for regression of human/mouse sgRNAs and classification of human/mouse sgRNA
### data_index = 3, 4 or 8
### when data_index = 3, pretype = 1
### when data_index = 4, pretype = 2
### when data_index = 7, pretype = 3
#    data_index1 = 3
#    pretype1 = 1
#    for featype1 in xrange(1, 4):
#        for phmmtype1 in xrange(0, 2):
#            data_index1, pretype1, featype1, phmmtype1=train_final_models(data_index1, pretype1, featype1, phmmtype1)
#            
#    data_index2 = 4
#    pretype2 = 2
#    for featype2 in xrange(1, 4):
#        for phmmtype2 in xrange(0, 2):
#            data_index2, pretype2, featype2, phmmtype1=train_final_models(data_index2, pretype2, featype2, phmmtype2)
#
#    data_index3 = 8
#    pretype3 = 3
#    for featype3 in xrange(1, 4):
#        for phmmtype3 in xrange(0, 2):
#            data_index3, pretype3, featype3, phmmtype3=train_final_models(data_index3, pretype3, featype3, phmmtype3)
################################################################################################################################    

################################ codes for final prediction ####################################################################
### the prediction receives 5 parameters:
###     1.filepath--the pasted or uploaded fasta sequence by user
###     2.filetype--determine the file type such as: filetype="annotated", where the file is downloaded from
###                ensembl database and contains all the gene information such exon, intron, transcripts and protein;
###                  or filetype="genome", where only the genome DNA sequence is provided (downloaded from database);
###                  or filetype="self defined", where the sequence is provided by users without any gene information
###     3.pretype--determine which specie is the gene from and which type of prediction: regression for human/mouse, zerbrafish; 
###                or classification for human/mouse
###     4.sgtype--determine the sgRNA cutting location: cut at exon (1) only or the whole gene(2)
###     5.phmmtype--whether use the phmm feature. As computing the phmm profiles is slow, if users hope to obtain the results
###                 faster, then set phmmtype=0 or phmmtype=1
#    
#    
#    
#    
#    
#    
#    
     filepath = sys.argv[1]
     filepath=filepath.replace('\\\\', '/')
     filepath=filepath.replace('\\', '/')
     filetype = sys.argv[2]
     species = sys.argv[3]
     pretype = int(sys.argv[4])
     sgtype = int(sys.argv[5])
     phmmtype = int(sys.argv[6])
     file_path, pre_scores=prediction_ce(filepath, filetype, species, pretype, sgtype, phmmtype)