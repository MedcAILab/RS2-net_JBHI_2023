from dataloader.Data_loader import *
from sklearn.metrics import roc_curve, auc, accuracy_score
from utils.BalanceBCEloss import bce2d
import copy
import sklearn.metrics
def validation_step(test_data_root, fold, model, criterion):
    model.eval()
    outPRED_mcs = torch.FloatTensor().cuda()
    label_list = torch.FloatTensor()
    lossValue = 0.0
    total = 0
    ID_list = []
    # precision_val_all = []
    # recall_val_all = []
    # f1_score_val_all = []
    for index, ID in enumerate(get_patient_ID(test_data_root, fold, mode='test')):

        data_test = DatasetGenerator(path=test_data_root, ID=ID, Aug=False, n_class=2, set_name='test')
        test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False,
                                                  num_workers=0, pin_memory=True, drop_last=True)
        total += len(test_loader)
        out = torch.FloatTensor([0, 0]).cuda()
        print('index: {}'.format(index))

        # precision_val = []
        # recall_val = []
        # f1_score_val = []
        for num, (imagesA, mask, label) in enumerate(test_loader):
            torch.set_grad_enabled(False)
            imagesA = imagesA.cuda()
            # mask = mask.cuda()
            # label_A, out_Seg = model(imagesA)
            label_A = model(imagesA)

            loss_x = criterion(label_A, label[0, :, :].cuda())
            # loss_Seg = criterion(torch.squeeze(out_Seg, dim=0), mask)
            # lossValue += loss_x.item() + loss_Seg.item()
            lossValue += loss_x.item()

            out = out + label_A
            print('num: {}'.format(num))

            # mask = mask.cpu().detach().numpy()
            # out_Seg = torch.squeeze(out_Seg, dim=0)
            # out_Seg = out_Seg.cpu().detach().numpy()
            # mask_pred_01 = np.float32(out_Seg > 0.5)
            # precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(mask.flatten(),
            #                                                                                  mask_pred_01.flatten(),
            #                                                                                  labels=[1])
            # precision_val.append(precision[0])
            # recall_val.append(recall[0])
            # f1_score_val.append(f1_score[0])

        label = label[0, :, :]
        out = out / len(test_loader)
        outPRED_mcs = torch.cat((outPRED_mcs, out), 0)
        label_list = torch.cat((label_list, label), 0)
        ID_list.append(ID)

        # precision_val_all.append(np.mean(precision_val))
        # recall_val_all.append(np.mean(recall_val))
        # f1_score_val_all.append(np.mean(f1_score_val))

    outPRED_mcs = outPRED_mcs.cpu().numpy()
    label_list = label_list.cpu().numpy()
    outPRED_mcs = outPRED_mcs[:, 1]
    label_list = label_list[:, 1]
    label_list = label_list.astype(np.int)
    fpr, tpr, thresholds = roc_curve(label_list, outPRED_mcs, pos_label=1)
    AUC_test = auc(fpr, tpr)

    Loss = lossValue / total

    return AUC_test, Loss, ID_list, label_list, outPRED_mcs #, precision_val_all, recall_val_all, f1_score_val_all

def externalData_step(external_data_root, fold, model, criterion):
    model.eval()
    outPRED_mcs = torch.FloatTensor().cuda()
    label_list = torch.FloatTensor()
    lossValue = 0.0
    total = 0
    ID_list = []
    # precision_test_all = []
    # recall_test_all = []
    # f1_score_test_all = []
    for index, ID in enumerate(get_patient_ID_ExternalData()):

        data_test = DatasetGenerator(path=external_data_root, ID=ID, Aug=False, n_class=2, set_name='test')
        test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False,
                                                  num_workers=0, pin_memory=True, drop_last=True)
        total += len(test_loader)
        out = torch.FloatTensor([0, 0]).cuda()
        print('index: {}'.format(index))

        # precision_test = []
        # recall_test = []
        # f1_score_test = []
        for num, (imagesA, mask, label) in enumerate(test_loader):
            torch.set_grad_enabled(False)
            imagesA = imagesA.cuda()
            # mask = mask.cuda()
            label_A = model(imagesA)

            loss_x = criterion(label_A, label[0, :, :].cuda())
            # loss_Seg = criterion(torch.squeeze(out_Seg, dim=0), mask)
            lossValue += loss_x.item()

            out = out + label_A
            print('num: {}'.format(num))

            # mask = mask.cpu().detach().numpy()
            # out_Seg = torch.squeeze(out_Seg, dim=0)
            # out_Seg = out_Seg.cpu().detach().numpy()
            # mask_pred_01 = np.float32(out_Seg > 0.5)
            # precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(mask.flatten(),
            #                                                                                  mask_pred_01.flatten(),
            #                                                                                  labels=[1])
            # precision_test.append(precision[0])
            # recall_test.append(recall[0])
            # f1_score_test.append(f1_score[0])

        label = label[0, :, :]
        out = out / len(test_loader)
        outPRED_mcs = torch.cat((outPRED_mcs, out), 0)
        label_list = torch.cat((label_list, label), 0)
        ID_list.append(ID)

        # precision_test_all.append(np.mean(precision_test))
        # recall_test_all.append(np.mean(recall_test))
        # f1_score_test_all.append(np.mean(f1_score_test))

    outPRED_mcs = outPRED_mcs.cpu().numpy()
    label_list = label_list.cpu().numpy()
    outPRED_mcs = outPRED_mcs[:, 1]
    label_list = label_list[:, 1]
    label_list = label_list.astype(np.int)
    fpr, tpr, thresholds = roc_curve(label_list, outPRED_mcs, pos_label=1)
    AUC_test = auc(fpr, tpr)

    Loss = lossValue / total

    return AUC_test, Loss, ID_list, label_list, outPRED_mcs #, precision_test_all, recall_test_all, f1_score_test_all








def validation_step_cla_seg(test_data_root, fold, model, criterion):
    model.eval()
    outPRED_mcs = torch.FloatTensor().cuda()
    label_list = torch.FloatTensor()
    lossValue = 0.0
    total = 0
    ID_list = []
    precision_val_all = []
    recall_val_all = []
    f1_score_val_all = []
    for index, ID in enumerate(get_patient_ID(test_data_root, fold, mode='test')):

        data_test = DatasetGenerator(path=test_data_root, ID=ID, Aug=False, n_class=2, set_name='test')
        test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False,
                                                  num_workers=0, pin_memory=True, drop_last=True)
        total += len(test_loader)
        out = torch.FloatTensor([0, 0]).cuda()
        print('index: {}'.format(index))

        precision_val = []
        recall_val = []
        f1_score_val = []
        for num, (imagesA, mask, label) in enumerate(test_loader):
            torch.set_grad_enabled(False)
            imagesA = imagesA.cuda()
            mask = mask.cuda()
            label_A, out_Seg= model(imagesA)

            loss_x = criterion(label_A, label[0, :, :].cuda())
            loss_Seg = criterion(torch.squeeze(out_Seg, dim=0), mask)
            lossValue += 1.0*loss_x.item()+0.25*loss_Seg.item()

            out = out + label_A
            print('num: {}'.format(num))



            mask = mask.cpu().detach().numpy()
            out_Seg = torch.squeeze(out_Seg, dim=0)
            out_Seg = out_Seg.cpu().detach().numpy()
            mask_pred_01 = np.float32(out_Seg > 0.5)
            precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(mask.flatten(),
                                                                                             mask_pred_01.flatten(),
                                                                                             labels=[1])
            precision_val.append(precision[0])
            recall_val.append(recall[0])
            f1_score_val.append(f1_score[0])

        label = label[0, :, :]
        out = out/len(test_loader)
        outPRED_mcs = torch.cat((outPRED_mcs, out), 0)
        label_list = torch.cat((label_list, label), 0)
        ID_list.append(ID)

        precision_val_all.append(np.mean(precision_val))
        recall_val_all.append(np.mean(recall_val))
        f1_score_val_all.append(np.mean(f1_score_val))

    outPRED_mcs= outPRED_mcs.cpu().numpy()
    label_list = label_list.cpu().numpy()
    outPRED_mcs = outPRED_mcs[:, 1]
    label_list = label_list[:, 1]
    label_list = label_list.astype(np.int)
    fpr, tpr, thresholds = roc_curve(label_list, outPRED_mcs, pos_label=1)
    AUC_test = auc(fpr, tpr)



    Loss = lossValue / total


    return AUC_test, Loss, ID_list, label_list, outPRED_mcs, precision_val_all, recall_val_all, f1_score_val_all


def externalData_step_cla_seg(external_data_root, fold, model, criterion):
    model.eval()
    outPRED_mcs = torch.FloatTensor().cuda()
    label_list = torch.FloatTensor()
    lossValue = 0.0
    total = 0
    ID_list = []
    precision_test_all = []
    recall_test_all = []
    f1_score_test_all = []
    for index, ID in enumerate(get_patient_ID_ExternalData()):

        data_test = DatasetGenerator(path=external_data_root, ID=ID, Aug=False, n_class=2, set_name='test')
        test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False,
                                                  num_workers=0, pin_memory=True, drop_last=True)
        total += len(test_loader)
        out = torch.FloatTensor([0, 0]).cuda()
        print('index: {}'.format(index))

        precision_test = []
        recall_test = []
        f1_score_test = []
        for num, (imagesA, mask, label) in enumerate(test_loader):
            torch.set_grad_enabled(False)
            imagesA = imagesA.cuda()
            mask = mask.cuda()
            label_A, out_Seg = model(imagesA)

            loss_x = criterion(label_A, label[0, :, :].cuda())
            loss_Seg = criterion(torch.squeeze(out_Seg, dim=0), mask)
            lossValue += 1.0*loss_x.item()+0.25*loss_Seg.item()

            out = out + label_A
            print('num: {}'.format(num))

            mask = mask.cpu().detach().numpy()
            out_Seg = torch.squeeze(out_Seg, dim=0)
            out_Seg = out_Seg.cpu().detach().numpy()
            mask_pred_01 = np.float32(out_Seg > 0.5)
            precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(mask.flatten(),
                                                                                             mask_pred_01.flatten(),
                                                                                             labels=[1])
            precision_test.append(precision[0])
            recall_test.append(recall[0])
            f1_score_test.append(f1_score[0])

        label = label[0, :, :]
        out = out / len(test_loader)
        outPRED_mcs = torch.cat((outPRED_mcs, out), 0)
        label_list = torch.cat((label_list, label), 0)
        ID_list.append(ID)

        precision_test_all.append(np.mean(precision_test))
        recall_test_all.append(np.mean(recall_test))
        f1_score_test_all.append(np.mean(f1_score_test))

    outPRED_mcs = outPRED_mcs.cpu().numpy()
    label_list = label_list.cpu().numpy()
    outPRED_mcs = outPRED_mcs[:, 1]
    label_list = label_list[:, 1]
    label_list = label_list.astype(np.int)
    fpr, tpr, thresholds = roc_curve(label_list, outPRED_mcs, pos_label=1)
    AUC_test = auc(fpr, tpr)

    Loss = lossValue / total

    return AUC_test, Loss, ID_list, label_list, outPRED_mcs, precision_test_all, recall_test_all, f1_score_test_all