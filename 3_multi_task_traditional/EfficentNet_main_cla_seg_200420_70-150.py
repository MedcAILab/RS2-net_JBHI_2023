from dataloader.Data_loader import *
import os
import torch
import shutil
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn
# from networks.efficientnet.efficient_origin import EfficientNet_b0
from utils.trainer_origin import *
from torch.optim import lr_scheduler
from torchsummary import summary
from utils.focal_loss import *
from utils.BCEfocalloss import *
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.validationAUC import *
from tensorboardX import SummaryWriter
from networks.resnest.resnest import resnest50, resnest50_cla_seg
# from networks.ResNest_aT.resnest import resnest50
'''70-150-外扩15 传统多任务分割分类'''

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# setup_seed(921)  # 921

# train_data_root = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP/PNETs_anchor_data_50-350_classify_center11111'
# vaild_data_root = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP/PNETs_anchor_data_50-350_classify_center11111'
# test_data_root = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP/PNETs_anchor_data_50-350_classify_center11111'
train_data_root = '/media/root/Elements/GP9/PNETs_anchor_data_70-150_classify_center1_plus15'
vaild_data_root = '/media/root/Elements/GP9/PNETs_anchor_data_70-150_classify_center1_plus15'
test_data_root = '/media/root/Elements/GP9/PNETs_anchor_data_70-150_classify_center2_plus15'

parser = argparse.ArgumentParser(description='imagenetCK19')
parser.add_argument('--model_dir', type=str, default='./result_origin/')
# parser.add_argument('--pre_model', type=bool, default=False)
# parser.add_argument('--save_dir', type=str, default='./result')
parser.add_argument('--save_model', type=str, default='ResNeSt50_')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16) # 8
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--cv', type=int, default=4)
parser.add_argument('--vfold', type=int, default=2)


args = parser.parse_args()
args.fold = 2
if args.fold==6:
    args.vfold =1
else:
    args.vfold = args.fold + 1
args.save_dir = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP1/LXY/pNENs_210420_cla_seg_70-150/' + str(args.fold) + '/model'
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# args.pre_model = False

best_metric = np.inf
best_auc = 0
best_syn = 0
best_iter = 0

# model = EfficientNet_b0(args.n_classes)
model = resnest50_cla_seg(pretrained=True)


model.to(device)
summary(model, (3, 256, 256))  # 224

# criterion = torch.nn.BCELoss(reduction='elementwise_mean')

# criterion = FocalLoss()
# criterion = BCEFocalLoss()
criterion = nn.BCELoss()
# criterion = 'Balance BCELoss'
# criterion = FocalLossV1()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

data_train = DatasetGenerator(path=train_data_root, Aug=True, n_class=args.n_classes, set_name='train', fold=args.fold, cv=args.cv)
train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size,
                                           shuffle=True, num_workers=8, pin_memory=True, drop_last=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
args.scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=args.epochs)

W_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP1/LXY/pNENs_210420_cla_seg_70-150/' + str(args.fold) + '/run'
if os.path.exists(W_path):
   shutil.rmtree(W_path)
writer = SummaryWriter(W_path)

for epoch in range(0, args.epochs):
    print('-----------------------------training stage-------------------------------------')
    train_loss, cla_loss, seg_loss = train_step_cls_seg(train_loader, model, epoch, optimizer, criterion, args)
    # train_loss = train_step_cls(train_loader, model, epoch, optimizer, criterion, args)
    writer.add_scalar('3_lr', args.scheduler.get_lr(), epoch)
    # writer.add_scalar('1_train_loss', train_loss, epoch)
    print('-----------------------------validing stage-------------------------------------')
    validation_auc, validation_Loss, valid_ID, valid_label, valid_score, val_precision, val_recall, val_f1 = validation_step_cla_seg(vaild_data_root, args.fold, model, criterion)
    # validation_auc, validation_Loss, valid_ID, valid_label, valid_score = validation_step(vaild_data_root, args.fold, model, criterion)
    # writer.add_scalar('2_validation_Loss', validation_Loss, epoch)
    # writer.add_scalar('4_validation_auc', validation_auc, epoch)
    print('Current auc: {}| Best auc: {} at epoch: {}'.format(validation_auc, best_auc, best_iter))
    # print('Current loss: {}| Best loss: {} at epoch: {}'.format(validation_Loss, best_metric, best_iter))
    print('-----------------------------Testing stage-------------------------------------')
    external_auc, external_Loss, external_ID, external_label, external_score, ex_precision, ex_recall, ex_f1 = externalData_step_cla_seg(test_data_root, args.fold, model, criterion)
    # external_auc, external_Loss, external_ID, external_label, external_score = externalData_step(test_data_root, args.fold, model, criterion)
    # writer.add_scalar('3_external_Loss', external_Loss, epoch)
    # writer.add_scalar('5_external_auc', external_auc, epoch)
    writer.add_scalars('1_loss', {'train_loss': train_loss,
                                  'validation_loss': validation_Loss,
                                  'external_loss': external_Loss}, epoch)
    writer.add_scalars('1_train_loss_cla_seg', {'cla_loss': cla_loss,
                                          'seg_loss': seg_loss}, epoch)
    writer.add_scalars('2_auc', {'validation_auc': validation_auc,
                                 'external_auc': external_auc}, epoch)
    writer.add_scalars('3_seg', {'validation_dsc': np.mean(val_f1),
                                 'external_dsc': np.mean(ex_f1)}, epoch)
    '''内部分类指标'''
    score_valid = valid_score
    label_valid = valid_label
    ID_valid = valid_ID
    result = {'ID': ID_valid, 'score': score_valid, 'label': label_valid}
    save_path = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP1/LXY/pNENs_210420_cla_seg_70-150/' \
                + str(args.fold) + '/V' + str(epoch + 1) + ' validation AUC ' + str(validation_auc) + '.csv'
    data_df = pd.DataFrame(result)
    data_df.to_csv(save_path)

    '''内部分割指标'''
    result_seg = {'ID': ID_valid, 'precison': val_precision, 'recall': val_recall, 'dsc': val_f1}
    save_path_seg = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP1/LXY/pNENs_210420_cla_seg_70-150/' \
                + str(args.fold) + '/V' + str(epoch + 1) + ' validation dsc ' + str(np.mean(val_f1)) + '.csv'
    data_df = pd.DataFrame(result_seg)
    data_df.to_csv(save_path_seg)

    '''外部分类指标'''
    score_external = external_score
    label_external = external_label
    ID_external = external_ID
    result2 = {'ID': ID_external, 'score': score_external, 'label': label_external}
    save_path2 = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP1/LXY/pNENs_210420_cla_seg_70-150/' \
                + str(args.fold) + '/E' + str(epoch + 1) + ' external AUC ' + str(external_auc) + '.csv'
    data_df2 = pd.DataFrame(result2)
    data_df2.to_csv(save_path2)

    '''外部分割指标'''
    result2_seg = {'ID': ID_external, 'precison': ex_precision, 'recall': ex_recall, 'dsc': ex_f1}
    save_path2_seg = '/media/root/83705360-882b-4f01-9201-5ba77b3d189b/20201112_Chia_Code_for classify/GP1/LXY/pNENs_210420_cla_seg_70-150/' \
                 + str(args.fold) + '/E' + str(epoch + 1) + ' external dsc ' + str(np.mean(ex_f1)) + '.csv'
    data_df2 = pd.DataFrame(result2_seg)
    data_df2.to_csv(save_path2_seg)

    if best_auc <= validation_auc:
        best_auc = validation_auc
        best_iter = epoch + 1

        model_save_file = os.path.join(args.save_dir, args.save_model + str(epoch) + '.tar')
        torch.save({'state_dict': model.state_dict(), 'best_auc': best_auc}, model_save_file)
        print('Model saved to %s' % model_save_file)


    writer.export_scalars_to_json(W_path + "/all_scalars.json")
    writer.close()

