import time
import torch
import sklearn
from progress.bar import Bar
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from skimage import transform
from .dice_loss import SoftDiceLoss
import torch.nn as nn
from utils.BalanceBCEloss import bce2d

def train_step(train_loader, model, epoch, optimizer, criterion, args):

    # switch to train mode
    model.train()
    epoch_loss = 0.0


    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)
        imagesA = imagesA.cuda()
        labels = labels.cuda()
        labels = labels[:,0,:]#.long()

        out_A,feature,feature2,feature3 = model(imagesA)

        # sigmoid 随时可注释掉
        out_A = nn.functional.sigmoid(out_A)
        feature2 = nn.functional.sigmoid(feature2)
        feature2 = feature2[:, 0, :, :]
        feature3 = nn.functional.sigmoid(feature3)
        feature3 = feature3[:, 0, :, :]

        # fc = model.fc.weight
        # fc1 = fc[0, :]
        # fc1 = torch.reshape(fc1, [1, 2048, 1, 1])
        # new_tensor = fc1 * feature

        # new_feature = new_tensor.cpu().data.numpy()
        # new_feature = np.array(new_feature, dtype=np.float64)
        # new2 = np.sum(new_feature, axis=1)


        # attation = (new2 - np.min(new2)) / (np.max(new2) - np.min(new2))
        # attation = transform.resize(attation,[16,224,224])



        mask = mask.cpu().data.numpy()
        mask = np.array(mask, dtype=np.float64)

        mask[mask>0]=1



        feature2 = feature2.cpu().data.numpy()
        feature2 = np.array(feature2, dtype=np.float64)
        feature3 = feature3.cpu().data.numpy()
        feature3 = np.array(feature3, dtype=np.float64)

        # attation = torch.from_numpy(attation)
        feature2 = torch.from_numpy(feature2)
        feature3 = torch.from_numpy(feature3)
        mask = torch.from_numpy(mask)


        dice_loss = SoftDiceLoss()

        dice1 = dice_loss(feature2,mask)
        dice2 = dice_loss(feature3, mask)
        # dice = dice_loss(attation,mask)



        loss_x = criterion(out_A, labels)

        # lossValue = loss_x

        # lossValue = loss_x * 0.6 + 0.4 * float(dice)
        lossValue = loss_x * 0.6 + 0.2 * float(dice1) + 0.2 * float(dice2)

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        args.sheudler.step()

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f}\n '
        bar.suffix = bar_str.format(step+1, iters_per_epoch, batch_time=batch_time*(iters_per_epoch-step)/60,
                                    loss=lossValue.item())
        bar.next()


    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return epoch_loss

def validation_step(val_loader, model, criterion):

    # switch to train mode
    model.eval()
    epoch_loss = 0
    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(val_loader):
        start_time = time.time()

        imagesA = imagesA.cuda()
        labels = labels.cuda()
        # labels = labels[:,0,0].long()
        labels = labels[:, 0, :]#.long()

        outputs,feature,feature2,feature3 = model(imagesA)
        # sigmoid 随时可注释掉
        outputs = nn.functional.sigmoid(outputs)

        with torch.no_grad():
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins\n'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    bar.finish()
    return epoch_loss

def train_step2(train_loader, model, epoch, optimizer, criterion, args):
    # switch to train mode
    model.train()
    epoch_loss = 0.0

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch + 1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)
        imagesA = imagesA.cuda()
        labels = labels.cuda()
        labels = labels[:, 0, :]  # .long()

        out_A = model(imagesA)

        loss_x = criterion(out_A, labels)

        lossValue = loss_x

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        args.sheudler.step()

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f}\n '
        bar.suffix = bar_str.format(step + 1, iters_per_epoch, batch_time=batch_time * (iters_per_epoch - step) / 60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch


    bar.finish()
    return epoch_loss

def validation_step2(val_loader, model, criterion):
    # switch to train mode
    model.eval()
    epoch_loss = 0
    # correct = torch.zeros(1).squeeze().cuda()
    # total = torch.zeros(1).squeeze().cuda()

    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(val_loader):
        start_time = time.time()

        imagesA = imagesA.cuda()
        labels = labels.cuda()
        # labels = labels[:,0,0].long()
        labels = labels[:, 0, :]  # .long()

        outputs = model(imagesA)
        # prediction = torch.argmax(outputs, 1)
        # y = torch.argmax(labels, 1)
        # correct += (prediction == y).sum().float()
        # total += len(labels)

        with torch.no_grad():
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins\n'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    # acc_str = (correct / total).cpu().detach().data.numpy()
    bar.finish()
    return epoch_loss

def train_step_seg(train_loader, model, epoch, optimizer, criterion, args):
    # switch to train mode
    model.train()
    epoch_loss = 0.0

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch + 1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)
        imagesA = imagesA.cuda()
        maskA = mask.cuda()
        labels = labels.cuda()
        labels = labels[:, 0, :]  # .long()

        out_A, label_B = model(imagesA)

        loss_x = criterion(out_A, maskA)

        lossValue = loss_x

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        args.sheudler.step()

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f}\n'
        bar.suffix = bar_str.format(step + 1, iters_per_epoch, batch_time=batch_time * (iters_per_epoch - step) / 60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch


    bar.finish()
    return epoch_loss

def validation_step_seg(val_loader, model, criterion):
    # switch to train mode
    model.eval()
    epoch_loss = 0
    # correct = torch.zeros(1).squeeze().cuda()
    # total = torch.zeros(1).squeeze().cuda()

    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(val_loader):
        start_time = time.time()

        imagesA = imagesA.cuda()
        labels = labels.cuda()
        maskA = mask.cuda()
        # labels = labels[:,0,0].long()
        labels = labels[:, 0, :]  # .long()

        outputs, label = model(imagesA)
        # prediction = torch.argmax(outputs, 1)
        # y = torch.argmax(labels, 1)
        # correct += (prediction == y).sum().float()
        # total += len(labels)

        with torch.no_grad():
            loss = criterion(outputs, maskA)
            epoch_loss += loss.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins\n'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    # acc_str = (correct / total).cpu().detach().data.numpy()
    bar.finish()
    return epoch_loss

def Normalization_seg(result, batchS):
    # result_mean = np.mean(result)
    # result = result + (0.5 - result_mean)
    for index in range(batchS):
        result_max, result_min = np.max(result[index, :, :]), np.min(result[index, :, :])
        result[index, :, :] = (result[index, :, :] - result_min) / (result_max - result_min)
    return result

def train_step_cls_seg(train_loader, model, epoch, optimizer, criterion, args):
    # switch to train mode
    model.train()
    epoch_loss = 0.0
    cla_loss = 0.0
    seg_loss1 = 0.0
    seg_loss = 0.0

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch + 1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False


    for step, (imagesA, mask, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)
        imagesA = imagesA.cuda()
        maskA = mask.cuda()
        labels = labels.cuda()
        labels = labels[:, 0, :]  # .long()
        # out_A, out_Seg = model(imagesA)
        OUT = model(imagesA)
        out_A = OUT[0]
        out_Seg1 = OUT[1]
        out_Seg = OUT[2]

        # loss_x = bce2d(out_A, labels)
        loss_x = criterion(out_A, labels)
        loss_seg1 = criterion(out_Seg1[:, 0, :, :], maskA)
        loss_seg = criterion(out_Seg[:, 0, :, :], maskA)
        # lossValue = 1.0 * loss_x + 0.5 * loss_seg1 + 0.5 * loss_seg

        if epoch<=50:
            a = 0 # 0
            b = 1.0 # 1.0
            lossValue = a * loss_x + b * loss_seg1
        elif (epoch >50) and (epoch<=100):
            a = 0.5
            b = 0.5
            lossValue = a * loss_x + b * loss_seg1
        elif epoch > 100:
            a = 1.0 # 1.0
            b = 0 # 0
            lossValue = a * loss_x + b * loss_seg1

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        args.scheduler.step()

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time

        cla_loss += loss_x.item()
        seg_loss1 += loss_seg1.item()
        seg_loss += loss_seg.item()
        mask = mask.cpu().detach().numpy()
        # out_Seg = torch.squeeze(out_Seg, dim=0)
        out_Seg = out_Seg[:, 0, :, :]
        out_Seg = out_Seg.cpu().detach().numpy()
        out_Seg_N = Normalization_seg(out_Seg, train_loader.batch_size)
        mask_pred_01 = np.float32(out_Seg_N > 0.5)
        precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(mask.flatten(),
                                                                                         mask_pred_01.flatten(),
                                                                                         labels=[1])

        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} | Loss_cla: {loss_cla:.4f} | Loss_seg1: {loss_seg1:.4f} |Loss_seg:{loss_seg:.4f} \n '
        bar.suffix = bar_str.format(step + 1, iters_per_epoch, batch_time=batch_time * (iters_per_epoch - step) / 60,
                                    loss=lossValue.item(), loss_cla=loss_x.item(), loss_seg1=loss_seg1.item(), loss_seg=loss_seg.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    cla_loss = cla_loss / iters_per_epoch
    seg_loss1 = seg_loss1 / iters_per_epoch
    seg_loss = seg_loss / iters_per_epoch

    bar.finish()
    return epoch_loss, cla_loss, seg_loss

def validation_step_cls_seg(val_loader, model, criterion):
    # switch to train mode
    model.eval()
    epoch_loss = 0
    # correct = torch.zeros(1).squeeze().cuda()
    # total = torch.zeros(1).squeeze().cuda()

    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(val_loader):
        start_time = time.time()

        imagesA = imagesA.cuda()
        mask = mask.cuda()
        labels = labels.cuda()
        # labels = labels[:,0,0].long()
        labels = labels[:, 0, :]  # .long()

        outputs, seg = model(imagesA)
        # prediction = torch.argmax(outputs, 1)
        # y = torch.argmax(labels, 1)
        # correct += (prediction == y).sum().float()
        # total += len(labels)

        with torch.no_grad():
            loss = bce2d(outputs, labels)
            loss_seg = bce2d(seg, mask)
            epoch_loss += loss.item() + loss_seg.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins\n'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    # acc_str = (correct / total).cpu().detach().data.numpy()
    bar.finish()
    return epoch_loss




def train_step_cls(train_loader, model, epoch, optimizer, criterion, args):
    # switch to train mode
    model.train()
    epoch_loss = 0.0
    cla_loss = 0.0

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch + 1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)
        imagesA = imagesA.cuda()
        # maskA = mask.cuda()
        labels = labels.cuda()
        labels = labels[:, 0, :]  # .long()
        out_A = model(imagesA)

        # loss_x = bce2d(out_A, labels)
        loss_x = criterion(out_A, labels)
        lossValue = loss_x

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        args.scheduler.step()
        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time

        cla_loss += loss_x.item()

        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} \n '
        bar.suffix = bar_str.format(step + 1, iters_per_epoch, batch_time=batch_time * (iters_per_epoch - step) / 60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return epoch_loss

# def validation_step_cls(val_loader, model, criterion):


