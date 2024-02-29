import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, recall_score

import matplotlib.pyplot as plt
from scipy import interp
import copy

def collect_file(file_path, match='.h5'):
    file_list = []
    for root, dir, files in os.walk(file_path):
        for f in files:
            if match.lower() in f.lower():
                file_list.append(os.path.join(root, f))
    return file_list

def Normalization(result):
    result_mean = np.mean(result)
    result = result + (0.5 - result_mean)
    result_max, result_min = np.max(result), np.min(result)
    result = (result - result_min) / (result_max - result_min)
    return result

def CalAUC(alllabels, allprobas):
    fpr, tpr, thresholds = roc_curve(alllabels, allprobas, pos_label=1)
    AUC_test = auc(fpr, tpr)
    distances = []
    for i in range(len(thresholds)):
        distances.append(((fpr[i]) ** 2 + (tpr[i] - 1) ** 2))
    a = distances.index(min(distances))
    threshold = thresholds[a]
    return AUC_test, fpr, tpr, threshold

def CalACC(allprobas,alllabels,threshold):
    label_pred = []
    label_true = alllabels
    for i in range(len(alllabels)):
        if allprobas[i] >= threshold:
            label_pred.append(1.0)
        elif allprobas[i] < threshold:
            label_pred.append(0.0)

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(alllabels)):
        if ((label_true[i] == 1) and (label_pred[i] == 1)):
            TP += 1
        elif ((label_true[i] == 0) and (label_pred[i] == 0)):
            TN += 1
        elif ((label_true[i] == 0) and (label_pred[i] == 1)):
            FP += 1
        elif ((label_true[i] == 1) and (label_pred[i] == 0)):
            FN += 1

    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TN + TP + FN + FP)
    sensitivity = TP / (TP + FN)
    return accuracy, sensitivity, specificity, TN, TP, FN, FP



# file_path = '/media/root/32686b5b-6d88-429d-b7cd-35208a8181c2/Graduation_P/CK19/external_resnet'
# # save_path = '/media/root/32686b5b-6d88-429d-b7cd-35208a8181c2/Graduation_P/CK19/CALout/'+ 'resnet_internal' + '.csv'
# save_path = '/media/root/32686b5b-6d88-429d-b7cd-35208a8181c2/Graduation_P/CK19/CALout/'+ 'resnet_external' + '.csv'
# result_file_list = collect_file(file_path, '.csv')
#
# result_list = []
# for index, file_path in enumerate(result_file_list):
#     file = pd.read_csv(file_path)
#     result = np.array(file .iloc[:, 1:])
#     result[:, -2] = Normalization(result[:, -2])
#     if index==0:
#         result_list = result
#     else:
#         result_list = np.concatenate((result_list, result), axis=0)
#
# alllabels = result_list[:, -1]
# alllabels = alllabels.astype(np.int)
# allprobas = result_list[:, -2]
#
#
# AUC_test, fpr, tpr, thresholds = CalAUC(alllabels, allprobas)
# accuracy, sensitivity, specificity, TN, TP, FN, FP = CalACC(allprobas, alllabels, thresholds)
# print('AUC : ', AUC_test)
# print('acc : ', accuracy)
# print('sen : ', sensitivity)
# print('spe : ', specificity)
#
# excel_index = {'ID': result_list[:, 0], 'score': result_list[:, -2], 'label': result_list[:, -1],
#                'AUC': AUC_test, 'acc': accuracy, 'sen': sensitivity, 'spe': specificity}
#
# data_df = pd.DataFrame(excel_index)
# data_df.to_csv(save_path)


model_name = 'single_resnest_0.827'
data_name = 'external'
save_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/70-150_外扩15/结果/' + model_name + '/'
# save_path_ck19 = save_path + model_name + '_' + data_name + '_.csv'
# save_path_mvi = save_path + model_name + '_' + data_name + '_mvi.csv'
save_path_mvi = save_path + model_name + '_' + data_name + '_.csv'

file_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/70-150_外扩15/概率值'
result_file_list = collect_file(file_path, '.csv')

result_list = []
for index, file_path in enumerate(result_file_list):
    file = pd.read_csv(file_path)
    result = np.array(file .iloc[:, 1:])
    # result[:, -4] = Normalization(result[:, -4])
    result[:, -2] = Normalization(result[:, -2])
    if index==0:
        result_list = result
    else:
        result_list = np.concatenate((result_list, result), axis=0)

# alllabels_ck19 = result_list[:, -3]
# alllabels_ck19 = alllabels_ck19.astype(np.int)
# allprobas_ck19 = result_list[:, -4]
#
# AUC_test, fpr, tpr, thresholds = CalAUC(alllabels_ck19, allprobas_ck19)
# accuracy, sensitivity, specificity, TN, TP, FN, FP = CalACC(allprobas_ck19, alllabels_ck19, thresholds)
# print('===CK19===')
# print('AUC : ', AUC_test)
# print('acc : ', accuracy)
# print('sen : ', sensitivity)
# print('spe : ', specificity)


alllabels_mvi = result_list[:, -1]
alllabels_mvi = alllabels_mvi.astype(np.int)
allprobas_mvi = result_list[:, -2]

AUC_test2, fpr2, tpr2, thresholds2 = CalAUC(alllabels_mvi, allprobas_mvi)
accuracy2, sensitivity2, specificity2, TN2, TP2, FN2, FP2 = CalACC(allprobas_mvi, alllabels_mvi, thresholds2)
print('===grade===')
print('AUC : ', AUC_test2)
print('acc : ', accuracy2)
print('sen : ', sensitivity2)
print('spe : ', specificity2)


# excel_index = {'ID': result_list[:, 0], 'score_ck19': result_list[:, 1], 'label_ck19': result_list[:, 2],
#                'AUC_ck19': AUC_test, 'acc_ck19': accuracy, 'sen_ck19': sensitivity, 'spe_ck19': specificity}
#
# data_df = pd.DataFrame(excel_index)
# data_df.to_csv(save_path_ck19)

# excel_index2 = {'ID': result_list[:, 0], 'score_mvi': result_list[:, 3], 'label_mvi': result_list[:, 4],
#                'AUC_mvi': AUC_test2, 'acc_mvi': accuracy2, 'sen_mvi': sensitivity2, 'spe_mvi': specificity2}
#
# data_df2 = pd.DataFrame(excel_index2)
# data_df2.to_csv(save_path_mvi)

excel_index2 = {'ID': result_list[:, 0], 'score': result_list[:, 1], 'label': result_list[:, 2],
               'AUC': AUC_test2, 'acc': accuracy2, 'sen': sensitivity2, 'spe': specificity2}

data_df2 = pd.DataFrame(excel_index2)
data_df2.to_csv(save_path_mvi)
