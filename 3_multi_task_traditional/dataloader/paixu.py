import pandas as pd
import numpy as np
import os
import xlrd
import copy
from xlutils.copy import copy as xl_copy
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from scipy import interp
import xlwt

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

'''读取每折的auc文件路径'''
val_fold_auc = []
for fold in range(1,5):
    file_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP/pNENs/00000/' + 'val_' + str(fold)
    result_file_list = collect_file(file_path, '.csv')
    result_file_list.sort(key=lambda x: -float(x.split(' ')[4][0:-4])) # 降序：-float(x.split(' ')[4][0:-4])
    val_fold_auc.append(result_file_list)

result_list = []
AUC_all = []
Thres = []
'''按顺序把上面每折的所有模型的预测概率和label累计到result_file_list_1,然后计算auc'''
# for i in range(99, -1, -1):
for i in range(0, 100, 1):
    for j in range(0, 4):
        # for index, file_path in enumerate(val_fold_auc[j][i]):
        file = pd.read_csv(val_fold_auc[j][i])
        result = np.array(file.iloc[:, 1:])
        result[:, -2] = Normalization(result[:, -2])
        if j == 0:
            result_file_list_1 = result
        else:
            result_file_list_1 = np.concatenate((result_file_list_1, result), axis=0)
        # print(0)
    print('完成一次交叉验证后的auc计算。')
    alllabels = result_file_list_1[:, -1]
    alllabels = alllabels.astype(np.int)
    allprobas = result_file_list_1[:, -2]
    fpr, tpr, thresholds = roc_curve(alllabels, allprobas, pos_label=1)
    AUC_test = auc(fpr, tpr)
    AUC = roc_auc_score(alllabels, allprobas)
    '''累计不同模型获得的auc'''
    AUC_all.append(AUC)
    Thres.append(thresholds)

result_auc = {'AUC': AUC_all, 'Thresholds:': Thres, }
    # save_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP/pNENs/00000/' \
    #             'Val_auc' + '.csv'
    # data_df = pd.DataFrame(result)
    # data_df.to_csv(save_path)

Temp = pd.DataFrame(AUC_all)
val_fold1 = pd.DataFrame(val_fold_auc[0])
val_fold2 = pd.DataFrame(val_fold_auc[1])
val_fold3 = pd.DataFrame(val_fold_auc[2])
val_fold4 = pd.DataFrame(val_fold_auc[3])
save_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP/pNENs/00000/val_auc.xlsx'
writer = pd.ExcelWriter(save_path)
Temp.to_excel(excel_writer=writer, sheet_name='auc', header=False)
val_fold1.to_excel(excel_writer=writer, sheet_name='fold1')
val_fold2.to_excel(excel_writer=writer, sheet_name='fold2')
val_fold3.to_excel(excel_writer=writer, sheet_name='fold3')
val_fold4.to_excel(excel_writer=writer, sheet_name='fold4')
writer.save()
writer.close()
# val_fold1.to_excel(save_path, sheet_name='1', header=False)
# save_path.save()
# val_fold2.to_excel(save_path, sheet_name='2', header=False)
# save_path.save()
# val_fold3.to_excel(save_path, sheet_name='3', header=False)
# save_path.save()
# val_fold4.to_excel(save_path, sheet_name='4', header=False)
# save_path.save()

print(1)