import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import pandas as pd

def CalAUC(alllabels, allprobas):
    fpr, tpr, thresholds = roc_curve(alllabels, allprobas, pos_label=1)
    AUC_test = auc(fpr,tpr)
    distances = []
    for i in range(len(thresholds)):
        distances.append(((fpr[i]) ** 2 + (tpr[i] - 1) ** 2))
    a = distances.index(min(distances))
    threshold = thresholds[a]
    return AUC_test,fpr, tpr, threshold

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
    return accuracy,sensitivity,specificity,TN,TP,FN,FP

def load_pro_calculate_auc(pathname1, num, all_result_path, type, sheet):
    df = pd.read_excel(pathname1, sheet_name=None)  # sheet_name 指定表单名的方式来读取
    Type = list(df.keys())  # sheet1\sheet2\...
    data_SVM = df[Type[sheet]]
    data_SVM_label = data_SVM.iloc[0:num, 3]
    data_SVM_label = data_SVM_label.T
    data_SVM_label = data_SVM_label.tolist()
    data_SVM_proa = data_SVM.iloc[0:num, 4]
    data_SVM_proa = data_SVM_proa.T
    data_SVM_proa = data_SVM_proa.tolist()
    AUC_test_SVM, fpr_SVM, tpr_SVM, threshold_SVM = CalAUC(data_SVM_label, data_SVM_proa)
    path_new1 = all_result_path
    DrawROC(fpr_SVM, tpr_SVM, AUC_test_SVM, path_new1, type, center)
    return fpr_SVM, tpr_SVM, AUC_test_SVM

def DrawROC(fpr,tpr,roc_auc,path_new,Type,clf):
    plt.figure()
    font = {'family':'WenQuanYi Zen Hei','weight':'normal','size':12}
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr, mean_tpr, 'r--',label='ROC (AUC = %0.3f)' % roc_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC', font)
    plt.xlabel('1-特异度', font)
    plt.ylabel('敏感度', font)
    plt.legend(loc='lower right',prop=font)
    plt.savefig(path_new+'/'+Type+'_'+clf+'.png')
    # plt.show()
def each_method_roc_AUC(path1,name1,roc_name='ROC'):
    pathname1 = path1 + name1
    df = pd.read_excel(pathname1, sheet_name=None)  # sheet_name 指定表单名的方式来读取
    Type = list(df.keys())  # sheet1\sheet2\...
    data_SVM = df[Type[0]]
    data_SVM_label = data_SVM.iloc[0:num, 3]
    data_SVM_label = data_SVM_label.T
    data_SVM_label = data_SVM_label.tolist()
    data_SVM_proa = data_SVM.iloc[0:num, 2]
    data_SVM_proa = data_SVM_proa.T
    data_SVM_proa = data_SVM_proa.tolist()
    AUC_test_SVM, fpr_SVM, tpr_SVM, threshold_SVM = CalAUC(data_SVM_label, data_SVM_proa)
    path_new1 = all_result_path
    DrawROC(fpr_SVM, tpr_SVM, AUC_test_SVM, path_new1, roc_name, center)
    return fpr_SVM, tpr_SVM, AUC_test_SVM
def Draw_gt_seg_ROC(fpr_SVM,tpr_SVM,roc_auc_SVM,fpr_LR,tpr_LR,roc_auc_LR,path_new,Type,clf):
    plt.figure()
    font = {'family':'WenQuanYi Zen Hei','weight':'normal','size':12}

    mean_fpr_SVM = np.linspace(0, 1, 100)
    mean_tpr_SVM = interp(mean_fpr_SVM, fpr_SVM, tpr_SVM)
    mean_tpr_SVM[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'r--',label='基于手动分割的LR (AUC = %0.3f)' % roc_auc_SVM, lw=2)

    mean_fpr_LR = np.linspace(0, 1, 100)
    mean_tpr_LR = interp(mean_fpr_LR, fpr_LR, tpr_LR)
    mean_tpr_LR[0] = 0.0
    plt.plot(mean_fpr_LR, mean_tpr_LR, 'g-', label='基于半自动分割的LR (AUC = %0.3f)' % roc_auc_LR, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC', font)

    plt.xlabel('1-特异度', font)
    plt.ylabel('敏感度', font)
    plt.legend(loc='lower right',prop=font)
    plt.savefig(path_new+'/'+Type+'_'+clf+'.png')
    plt.show()
def Draw_single_multi_ROC(fpr_SVM,tpr_SVM,roc_auc_SVM,fpr_LR,tpr_LR,roc_auc_LR,path_new,Type,clf):
    plt.figure()
    font = {'family':'WenQuanYi Zen Hei','weight':'normal','size':12}

    mean_fpr_SVM = np.linspace(0, 1, 100)
    mean_tpr_SVM = interp(mean_fpr_SVM, fpr_SVM, tpr_SVM)
    mean_tpr_SVM[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'r--',label='单任务网络 (AUC = %0.3f)' % roc_auc_SVM, lw=2)

    mean_fpr_LR = np.linspace(0, 1, 100)
    mean_tpr_LR = interp(mean_fpr_LR, fpr_LR, tpr_LR)
    mean_tpr_LR[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_LR, mean_tpr_LR, 'g-', label='多任务网络 (AUC = %0.3f)' % roc_auc_LR, lw=2)



    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC curve', font)


    plt.xlabel('1-特异度', font)  # False Positive Rate
    plt.ylabel('敏感度', font)  # True Positive Rate
    plt.legend(loc='lower right',prop=font)
    plt.savefig(path_new+'/'+Type+'_'+clf+'.png')
    plt.show()
def Draw_single_multi_multi2_ROC(fpr_SVM,tpr_SVM,roc_auc_SVM,fpr_LR,tpr_LR,roc_auc_LR,fpr_RF,tpr_RF,roc_auc_RF,path_new,Type,clf):
    plt.figure()
    font = {'family':'WenQuanYi Zen Hei','weight':'normal','size':12}

    mean_fpr_SVM = np.linspace(0, 1, 100)
    mean_tpr_SVM = interp(mean_fpr_SVM, fpr_SVM, tpr_SVM)
    mean_tpr_SVM[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'b--',label='DenseNet (AUC = %0.3f)' % roc_auc_SVM, lw=2)
    # plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'b--', label='单任务深度学习 (AUC = %0.3f)' % roc_auc_SVM, lw=2)

    mean_fpr_LR = np.linspace(0, 1, 100)
    mean_tpr_LR = interp(mean_fpr_LR, fpr_LR, tpr_LR)
    mean_tpr_LR[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_LR, mean_tpr_LR, 'g--', label='SKNet (AUC = %0.3f)' % roc_auc_LR, lw=2)
    # plt.plot(mean_fpr_LR, mean_tpr_LR, 'g--', label='多任务深度学习 (AUC = %0.3f)' % roc_auc_LR, lw=2)

    mean_fpr_RF = np.linspace(0, 1, 100)
    mean_tpr_RF = interp(mean_fpr_RF, fpr_RF, tpr_RF)
    mean_tpr_RF[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_RF, mean_tpr_RF, 'r--', label='ResNeSt (AUC = %0.3f)' % roc_auc_RF, lw=2)
    # plt.plot(mean_fpr_RF, mean_tpr_RF, 'r--', label='改进的多任务深度学习 (AUC = %0.3f)' % roc_auc_RF, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # plt.title('ROC curve', font)

    plt.xlabel('1-特异度', font)  # False Positive Rate
    plt.ylabel('敏感度', font)  # True Positive Rate
    plt.legend(loc='lower right',prop=font)
    plt.savefig(path_new+'/'+Type+'_'+clf+'.png')
    plt.show()
def Draw_DL_vs_4RD_ROC(fpr_GBDT,tpr_GBDT,roc_auc_GBDT,fpr_RF,tpr_RF,roc_auc_RF,fpr_SVM,tpr_SVM,roc_auc_SVM,fpr_LR,tpr_LR,roc_auc_LR,fpr_2,tpr_2,roc_auc_2,path_new,Type,clf):
    plt.figure()
    font = {'family':'WenQuanYi Zen Hei','weight':'normal','size':12}

    mean_fpr_GBDT = np.linspace(0, 1, 100)
    mean_tpr_GBDT = interp(mean_fpr_GBDT, fpr_GBDT, tpr_GBDT)
    mean_tpr_GBDT[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_GBDT, mean_tpr_GBDT, 'g--', label='基于半自动分割的GBDT (AUC = %0.3f)' % roc_auc_GBDT, lw=2)

    mean_fpr_RF = np.linspace(0, 1, 100)
    mean_tpr_RF = interp(mean_fpr_RF, fpr_RF, tpr_RF)
    mean_tpr_RF[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_RF, mean_tpr_RF, 'y--', label='基于半自动分割的RF (AUC = %0.3f)' % roc_auc_RF, lw=2)

    mean_fpr_SVM = np.linspace(0, 1, 100)
    mean_tpr_SVM = interp(mean_fpr_SVM, fpr_SVM, tpr_SVM)
    mean_tpr_SVM[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'm--',label='基于半自动分割的SVM (AUC = %0.3f)' % roc_auc_SVM, lw=2)

    mean_fpr_LR = np.linspace(0, 1, 100)
    mean_tpr_LR = interp(mean_fpr_LR, fpr_LR, tpr_LR)
    mean_tpr_LR[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_LR, mean_tpr_LR, 'b--', label='基于半自动分割的LR (AUC = %0.3f)' % roc_auc_LR, lw=2)

    # mean_fpr_2 = np.linspace(0, 1, 100)
    # mean_tpr_2 = interp(mean_fpr_2, fpr_2, tpr_2)
    # mean_tpr_2[0] = 0.0
    # plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    # plt.plot(mean_fpr_2, mean_tpr_2, 'r--', label='单任务深度学习 (AUC = %0.3f)' % roc_auc_2, lw=2)

    mean_fpr_2 = np.linspace(0, 1, 100)
    mean_tpr_2 = interp(mean_fpr_2, fpr_2, tpr_2)
    mean_tpr_2[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_2, mean_tpr_2, 'r--', label='改进后的多任务深度学习 (AUC = %0.3f)' % roc_auc_2, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # plt.title('ROC curve', font)

    plt.xlabel('1-特异度', font)  # False Positive Rate
    plt.ylabel('敏感度', font)  # True Positive Rate
    plt.legend(loc='lower right',prop=font)
    plt.savefig(path_new+'/'+Type+'_'+clf+'.png')
    plt.show()

def one_ROC(path1, name1, num, type='DenseNet', center=''):
    pathname1 = path1 + name1
    df = pd.read_excel(pathname1, sheet_name=None)  # sheet_name 指定表单名的方式来读取
    Type = list(df.keys())  # sheet1\sheet2\...
    data_SVM = df[Type[0]]
    data_SVM_label = data_SVM.iloc[0:num, 3]
    data_SVM_label = data_SVM_label.T
    data_SVM_label = data_SVM_label.tolist()
    data_SVM_proa = data_SVM.iloc[0:num, 2]
    data_SVM_proa = data_SVM_proa.T
    data_SVM_proa = data_SVM_proa.tolist()
    AUC_test_SVM, fpr_SVM, tpr_SVM, threshold_SVM = CalAUC(data_SVM_label, data_SVM_proa)
    path_new1 = all_result_path
    DrawROC(fpr_SVM, tpr_SVM, AUC_test_SVM, path_new1, type, center)
    return fpr_SVM, tpr_SVM, AUC_test_SVM

def Draw2ROC(fpr_SVM,tpr_SVM,roc_auc_SVM,fpr_LR,tpr_LR,roc_auc_LR,path_new,Type,clf):
    plt.figure()
    font = {'family':'Times New Roman','weight':'normal','size':12}

    mean_fpr_SVM = np.linspace(0, 1, 100)
    mean_tpr_SVM = interp(mean_fpr_SVM, fpr_SVM, tpr_SVM)
    mean_tpr_SVM[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'r--',label='SVM ROC (AUC = %0.2f)' % roc_auc_SVM, lw=1)

    mean_fpr_LR = np.linspace(0, 1, 100)
    mean_tpr_LR = interp(mean_fpr_LR, fpr_LR, tpr_LR)
    mean_tpr_LR[0] = 0.0
    plt.plot(mean_fpr_LR, mean_tpr_LR, 'g-', label='LR ROC (AUC = %0.2f)' % roc_auc_LR, lw=1)



    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC curve', font)


    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)
    plt.legend(loc='lower right',prop=font)
    plt.savefig(path_new+'/'+Type+'_'+clf+'.png')
    plt.show()

def DrawAllROC(fpr_SVM, tpr_SVM, roc_auc_SVM, fpr_LR, tpr_LR, roc_auc_LR, fpr_RF, tpr_RF, roc_auc_RF, fpr_GBDT,
               tpr_GBDT, roc_auc_GBDT, path_new, Type, clf):
    plt.figure()
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

    mean_fpr_SVM = np.linspace(0, 1, 100)
    mean_tpr_SVM = interp(mean_fpr_SVM, fpr_SVM, tpr_SVM)
    mean_tpr_SVM[0] = 0.0
    plt.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7))
    plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'r--', label='SVM ROC (AUC = %0.2f)' % roc_auc_SVM, lw=1)

    mean_fpr_LR = np.linspace(0, 1, 100)
    mean_tpr_LR = interp(mean_fpr_LR, fpr_LR, tpr_LR)
    mean_tpr_LR[0] = 0.0
    plt.plot(mean_fpr_LR, mean_tpr_LR, 'g-', label='LR ROC (AUC = %0.2f)' % roc_auc_LR, lw=1)

    mean_fpr_RF = np.linspace(0, 1, 100)
    mean_tpr_RF = interp(mean_fpr_RF, fpr_RF, tpr_RF)
    mean_tpr_RF[0] = 0.0
    plt.plot(mean_fpr_RF, mean_tpr_RF, 'b-.', label='RF ROC (AUC = %0.2f)' % roc_auc_RF, lw=1)

    mean_fpr_GBDT = np.linspace(0, 1, 100)
    mean_tpr_GBDT = interp(mean_fpr_GBDT, fpr_GBDT, tpr_GBDT)
    mean_tpr_GBDT[0] = 0.0
    plt.plot(mean_fpr_GBDT, mean_tpr_GBDT, 'y:', label='GBDT ROC (AUC = %0.2f)' % roc_auc_GBDT, lw=1)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC curve', font)

    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)
    plt.legend(loc='lower right', prop=font)
    plt.savefig(path_new + '/' + Type + '_' + clf + '.png')
    plt.show()




if __name__ == "__main__":
    ###########################################################################################################################################
    '''radiomics'''
    # all_result_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/radiomics/'
    # center = 'external'
    # if center == 'internal':
    #     num = 80
    #     name1 = '内部数据-金标准.xlsx'
    #     name2 = '内部数据-分割结果.xlsx'
    # elif center == 'external':
    #     num = 21
    #     name1 = '外部数据-金标准.xlsx'
    #     name2 = '外部数据-分割结果.xlsx'
    #
    # pathname1 = all_result_path + name1
    # fpr_seg, tpr_seg, AUC_test_seg = load_pro_calculate_auc(pathname1, num, all_result_path, type='SVM_gt', sheet=1)
    # pathname2 = all_result_path + name2
    # fpr_gt, tpr_gt, AUC_test_gt = load_pro_calculate_auc(pathname2, num, all_result_path, type='SVM_seg', sheet=1)
    # path_new = all_result_path
    # Draw_gt_seg_ROC(fpr_seg, tpr_seg, AUC_test_seg,fpr_gt, tpr_gt, AUC_test_gt, path_new,Type='SVM',clf=center)
    # print(0)
    ###########################################################################################################################################
    '''single-multi-task-cls-seg'''
    # all_result_path2 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/'
    # center = 'external'
    # if center == 'internal':
    #     num = 80
    #     name1 = 'cla_seg_DL_probability.xlsx'
    #     name2 = 'cla_seg_DL_probability.xlsx'
    # elif center == 'external':
    #     num = 18
    #     name1 = 'E14 external AUC 0.8153846153846154.xlsx'
    #     name2 = 'E12 external AUC 0.8461538461538463.xlsx'
    # path1 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP/pNENs_201223_cla/trainmodel-AUC=0.82843-fold2-0.81538（这个）/'
    # pathname1 = path1 + name1
    # fpr_single, tpr_single, AUC_test_single = load_pro_calculate_auc(pathname1, num, all_result_path2, type='multi')
    # path2 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP1/LXY/pNENs_201217_cla_seg/trainmodel-AUC=0.83149-fold4-0.84615（这个）/'
    # pathname2 = path2 + name2
    # fpr_multi, tpr_multi, AUC_test_multi = load_pro_calculate_auc(pathname2, num, all_result_path2, type='single')
    # path_new = all_result_path2
    # Draw_single_multi_ROC(fpr_single, tpr_single, AUC_test_single, fpr_multi, tpr_multi, AUC_test_multi, path_new,Type='single_multi',clf=center)
    # print(0)
    '''改了LR外部结果所以重新画了图'''
    # all_result_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/radiomics/'
    # center = 'internal'
    # if center == 'internal':
    #     num = 80
    #     name1 = '内部数据-金标准.xlsx'
    #     name2 = '内部数据-分割结果.xlsx'
    # elif center == 'external':
    #     num = 21
    #     name1 = 'LR_210228_result_GT.xlsx'
    #     name2 = 'LR_210228_result_SEG.xlsx'
    #
    # pathname1 = all_result_path + name1
    # fpr_seg, tpr_seg, AUC_test_seg = load_pro_calculate_auc(pathname1, num, all_result_path, type='LR_gt', sheet=2)
    # pathname2 = all_result_path + name2
    # fpr_gt, tpr_gt, AUC_test_gt = load_pro_calculate_auc(pathname2, num, all_result_path, type='LR_seg', sheet=2)
    # path_new = all_result_path
    # Draw_gt_seg_ROC(fpr_seg, tpr_seg, AUC_test_seg, fpr_gt, tpr_gt, AUC_test_gt, path_new,Type='LR',clf=center)
    # print(0)

#######################################################################################################################################################

    # pathname = '/media/root/32686b5b-6d88-429d-b7cd-35208a8181c2/feature_extraction2/Ng=64_Uniform_gt/LassoCV_Tx_Sp_gt/FeatureSelect.xlsx'
    # df = pd.read_excel(pathname, sheet_name=None)  # sheet_name 指定表单名的方式来读取
    # Type = list(df.keys())  # sheet1\sheet2\...
    # data_SVM = df[Type[1]]
    # data_SVM_label = data_SVM.iloc[0:80, 3]
    # data_SVM_label = data_SVM_label.T
    # data_SVM_label = data_SVM_label.tolist()
    # data_SVM_proa = data_SVM.iloc[0:80, 4]
    # data_SVM_proa = data_SVM_proa.T
    # data_SVM_proa = data_SVM_proa.tolist()
    # AUC_test_SVM, fpr_SVM, tpr_SVM, threshold_SVM = CalAUC(data_SVM_label, data_SVM_proa)
    #
    # data_LR = df[Type[2]]
    # data_LR_label = data_LR.iloc[0:80, 3]
    # data_LR_label = data_LR_label.T
    # data_LR_label = data_LR_label.tolist()
    # data_LR_proa = data_LR.iloc[0:80, 4]
    # data_LR_proa = data_LR_proa.T
    # data_LR_proa = data_LR_proa.tolist()
    # AUC_test_LR, fpr_LR, tpr_LR, threshold_LR = CalAUC(data_LR_label, data_LR_proa)
    #
    # data_RF = df[Type[3]]
    # data_RF_label = data_RF.iloc[0:80, 3]
    # data_RF_label = data_RF_label.T
    # data_RF_label = data_RF_label.tolist()
    # data_RF_proa = data_RF.iloc[0:80, 4]
    # data_RF_proa = data_RF_proa.T
    # data_RF_proa = data_RF_proa.tolist()
    # AUC_test_RF, fpr_RF, tpr_RF, threshold_RF = CalAUC(data_RF_label, data_RF_proa)
    #
    # data_GBDT = df[Type[4]]
    # data_GBDT_label = data_GBDT.iloc[0:80, 3]
    # data_GBDT_label = data_GBDT_label.T
    # data_GBDT_label = data_GBDT_label.tolist()
    # data_GBDT_proa = data_GBDT.iloc[0:80, 4]
    # data_GBDT_proa = data_GBDT_proa.T
    # data_GBDT_proa = data_GBDT_proa.tolist()
    # AUC_test_GBDT, fpr_GBDT, tpr_GBDT, threshold_GBDT = CalAUC(data_GBDT_label, data_GBDT_proa)
    #
    # path_new = '/root/Desktop/5173'
    # # Draw2ROC(fpr_SVM, tpr_SVM, AUC_test_SVM, fpr_LR, tpr_LR, AUC_test_LR, path_new, 'GT', 'ROC')
    # DrawAllROC(fpr_SVM, tpr_SVM, AUC_test_SVM, fpr_LR, tpr_LR, AUC_test_LR, fpr_RF, tpr_RF, AUC_test_RF, fpr_GBDT,
    #            tpr_GBDT, AUC_test_GBDT, path_new, 'SEG', 'ROC')

#######################################################################################################################################################
    '''单任务/多任务深度学习 vs 影像组学'''
    # all_result_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/'
    # center = 'external'
    # if center == 'internal':
    #     num = 80
    #     # name1 = 'cla_seg_DL_probability.xlsx'
    #     # name2 = 'cla_seg_DL_probability.xlsx'
    #     # name3 = 'cla_seg_DL_probability.xlsx'
    #     name1 = 'cla_seg_DL_probability.xlsx'
    #     name2 = '内部数据-分割结果.xlsx'
    #     name3 = '内部数据-分割结果.xlsx'
    #     name4 = '内部数据-分割结果.xlsx'
    #     name5 = '内部数据-分割结果.xlsx'
    # elif center == 'external':
    #     num =21 # 21 18
    #     # name1 = 'E72 external AUC 0.8.xlsx'
    #     # name2 = 'E2 external AUC 0.8.xlsx'
    #     # name3 = 'E14 external AUC 0.8153846153846154.xlsx'
    #
    #     # name1 = 'E14 external AUC 0.8153846153846154.xlsx' # 单任务
    #     # name1 = 'E15 external AUC 0.8194444444444445.xlsx' # 单任务-70-150
    #     # name1 = 'E93 external AUC 0.875.xlsx' # 改进多任务-70-150
    #     name1 = 'E103 external AUC 0.8749999999999999.xlsx' # 改进多任务-70-150
    #     name2 = 'LR_210228_result_SEG.xlsx'
    #     name3 = '外部数据-分割结果.xlsx'
    #     name4 = '外部数据-分割结果.xlsx'
    #     name5 = '外部数据-分割结果.xlsx'
    # # path1 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-densenet/'
    # # fpr_1, tpr_1, AUC_test_1 = each_method_roc_AUC(path1, name1, roc_name='densenet')
    # # path2 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-sknet/'
    # # fpr_2, tpr_2, AUC_test_2 = each_method_roc_AUC(path1, name1, roc_name='sknet')
    # # path3 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-resnest/'
    # # fpr_3, tpr_3, AUC_test_3 = each_method_roc_AUC(path1, name1, roc_name='resnest')
    #
    # # path1 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-resnest/'
    # # fpr_1, tpr_1, AUC_test_1 = each_method_roc_AUC(path1, name1, roc_name='resnest')
    # path1 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/多任务-改进_70-150_外扩15/trainmodel-AUC=0.95343-fold2-0.87499（这个）/'
    # fpr_1, tpr_1, AUC_test_1 = each_method_roc_AUC(path1, name1, roc_name='resnest')
    # path2 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/radiomics/'
    # fpr_2, tpr_2, AUC_test_2 = load_pro_calculate_auc(path2 + name2, num, all_result_path, type='LR_seg', sheet=2)
    # path3 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/radiomics/'
    # fpr_3, tpr_3, AUC_test_3 = load_pro_calculate_auc(path3 + name3, num, all_result_path, type='SVM_seg', sheet=1)
    # path4 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/radiomics/'
    # fpr_4, tpr_4, AUC_test_4 = load_pro_calculate_auc(path4 + name4, num, all_result_path, type='RF_seg', sheet=3)
    # path5 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/radiomics/'
    # fpr_5, tpr_5, AUC_test_5 = load_pro_calculate_auc(path5 + name5, num, all_result_path, type='GBDT_seg', sheet=4)
    # Draw_DL_vs_4RD_ROC(fpr_5,tpr_5,AUC_test_5,fpr_4,tpr_4,AUC_test_4,fpr_3,tpr_3,AUC_test_3,fpr_2,tpr_2,AUC_test_2,fpr_1,tpr_1,AUC_test_1,all_result_path,'multiDL_vs_RD',center)

#######################################################################################################################################################
    '''单任务 不同网络对比'''
    all_result_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/'
    center = 'internal'
    if center == 'internal':
        num = 80
    elif center == 'external':
        num = 18 # 21 18
    path1 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-densenet_70-150_外扩15/trainmodel-AUC=0.79902-fold1-0.80556/'
    name1 = 'cla_seg_DL_probability.xlsx'
    # name1 = 'E11 external AUC 0.8055555555555556.xlsx'
    pathname1 = path1 + name1
    df = pd.read_excel(pathname1, sheet_name=None)  # sheet_name 指定表单名的方式来读取
    Type = list(df.keys())  # sheet1\sheet2\...
    data_SVM = df[Type[0]]
    data_SVM_label = data_SVM.iloc[0:num, 3]
    data_SVM_label = data_SVM_label.T
    data_SVM_label = data_SVM_label.tolist()
    data_SVM_proa = data_SVM.iloc[0:num, 2]
    data_SVM_proa = data_SVM_proa.T
    data_SVM_proa = data_SVM_proa.tolist()
    AUC_test_SVM, fpr_SVM, tpr_SVM, threshold_SVM = CalAUC(data_SVM_label, data_SVM_proa)
    path_new1 = all_result_path
    DrawROC(fpr_SVM, tpr_SVM, AUC_test_SVM, path_new1, 'DenseNet', center)
# ----------------------------------------------------------------------------------------------------------------------
    path2 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-sknet_70-150_外扩15/trainmodel-AUC=0.80392-fold2-0.80556（这个）/'
    name2 = 'cla_seg_DL_probability.xlsx'
    # name2 = 'E1 external AUC 0.8055555555555556.xlsx'
    pathname2 = path2 + name2
    df2 = pd.read_excel(pathname2, sheet_name=None)  # sheet_name 指定表单名的方式来读取
    Type = list(df2.keys())  # sheet1\sheet2\...
    data_LR = df2[Type[0]]
    data_LR_label = data_LR.iloc[0:num, 3]
    data_LR_label = data_LR_label.T
    data_LR_label = data_LR_label.tolist()
    data_LR_proa = data_LR.iloc[0:num, 2]
    data_LR_proa = data_LR_proa.T
    data_LR_proa = data_LR_proa.tolist()
    AUC_test_LR, fpr_LR, tpr_LR, threshold_LR = CalAUC(data_LR_label, data_LR_proa)
    path_new2 = all_result_path
    DrawROC(fpr_LR, tpr_LR, AUC_test_LR, path_new2, 'SKNet', center)
# ----------------------------------------------------------------------------------------------------------------------
    path3 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-resnest_70-150_外扩15/trainmodel-AUC=0.82598-fold1-0.81944/'
    name3 = 'cla_seg_DL_probability.xlsx'
    # name3 = 'E15 external AUC 0.8194444444444445.xlsx'
    pathname3 = path3 + name3
    df2 = pd.read_excel(pathname3, sheet_name=None)  # sheet_name 指定表单名的方式来读取
    Type = list(df2.keys())  # sheet1\sheet2\...
    data_RF = df2[Type[0]]
    data_RF_label = data_RF.iloc[0:num, 3]
    data_RF_label = data_RF_label.T
    data_RF_label = data_RF_label.tolist()
    data_RF_proa = data_RF.iloc[0:num, 2]
    data_RF_proa = data_RF_proa.T
    data_RF_proa = data_RF_proa.tolist()
    AUC_test_RF, fpr_RF, tpr_RF, threshold_RF = CalAUC(data_RF_label, data_RF_proa)
    path_new3 = all_result_path
    DrawROC(fpr_RF, tpr_RF, AUC_test_RF, path_new3, 'ResNeSt', center)
# ----------------------------------------------------------------------------------------------------------------------
    path_new = all_result_path
    # 注意修改Draw_single_multi_multi2_ROC里面的图例
    # Draw_single_multi_ROC(fpr_SVM, tpr_SVM, AUC_test_SVM, fpr_LR, tpr_LR, AUC_test_LR, path_new, 'multi_vs_single', center)
    Draw_single_multi_multi2_ROC(fpr_SVM, tpr_SVM, AUC_test_SVM, fpr_LR, tpr_LR, AUC_test_LR, fpr_RF, tpr_RF, AUC_test_RF, path_new, 'densenet_vs_sknet_vs_resnest', center)

#######################################################################################################################################################
    '''单任务 vs 传统多任务 vs 改进多任务'''
    # all_result_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/'
    # center = 'external'
    # if center == 'internal':
    #     num = 80
    #     name1 = 'cla_seg_DL_probability.xlsx'
    #     name2 = 'cla_seg_DL_probability.xlsx'
    #     name3 = 'cla_seg_DL_probability.xlsx'
    # elif center == 'external':
    #     num = 18  # 21 18
    #     name1 = 'E15 external AUC 0.8194444444444445.xlsx'
    #     name2 = 'E37 external AUC 0.8472222222222223.xlsx'
    #     # name2 = 'E3 external AUC 0.8611111111111112.xlsx'
    #     # name3 = 'E103 external AUC 0.8749999999999999.xlsx'
    #     name3 = 'E93 external AUC 0.875.xlsx'
    # path1 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/单任务-resnest_70-150_外扩15/trainmodel-AUC=0.82659-fold1-0.81944（这个）/'
    # path2 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/多任务-传统_70-150_外扩15/trainmodel-AUC=0.83282-fold5-0.84722（这个）/'
    # path3 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/result/多任务-改进_70-150_外扩15/trainmodel-AUC=0.88664-fold1-0.87500/'
    #
    # fpr_single, tpr_single, AUC_test_single = one_ROC(path1, name1, num, type='Single', center=center)
    # fpr_multi1, tpr_multi1, AUC_test_multi1 = one_ROC(path2, name2, num, type='Multi', center=center)
    # fpr_multi2, tpr_multi2, AUC_test_multi2 = one_ROC(path3, name3, num, type='Multi2', center=center)
    #
    #
    # path_new = all_result_path
    # Draw_single_multi_multi2_ROC(fpr_single, tpr_single, AUC_test_single, fpr_multi1, tpr_multi1, AUC_test_multi1, fpr_multi2, tpr_multi2, AUC_test_multi2, path_new, 'Single_vs_Multi_vs_Multi2', center)