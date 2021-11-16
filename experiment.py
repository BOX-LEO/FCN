import FCN32
import FCN16
import numpy as np
import torch
import shutil
import os, errno
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import KittiDataset as K


def mv_image(org, dis, images):
    for image in images:
        try:
            os.makedirs(dis, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        original = org + image
        target = dis + image
        shutil.copyfile(original, target)


def split_dataset(path, mode='image', valrate=0.15, testrate=0.15):
    if valrate + testrate >= 1:
        print('require valrate+testrate<1')
    else:
        train_path = './data/' + mode + '/train/'
        val_path = './data/' + mode + '/val/'
        test_path = './data/' + mode + '/test/'
        images = os.listdir(path)
        sorted(images)
        size = len(images)

        val_size = int(size * valrate)
        test_size = int(size * testrate)
        val_images = images[0:val_size]
        test_image = images[val_size:val_size + test_size]
        train_image = images[val_size + test_size:]
        mv_image(path, train_path, train_image)
        mv_image(path, val_path, val_images)
        mv_image(path, test_path, test_image)


def pIoU(pred_mask, mask, n_class=34):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask)
        pred_mask = torch.argmax(pred_mask, dim=1)

        TP_per_class = []
        FP_per_class = []
        FN_per_class = []

        for c in range(0, n_class + 1):
            true_pred = pred_mask == c
            true_label = mask == c
            if true_label.long().sum().item() == 0:  # no exist label in this loop
                TP_per_class.append(np.nan)
                FP_per_class.append(np.nan)
                FN_per_class.append(np.nan)
            else:
                TP = torch.logical_and(true_pred,
                                       true_label).sum().float().item()
                FP = torch.logical_and(torch.logical_not(true_pred),
                                       true_label).sum().float().item()
                FN = torch.logical_and(true_pred,
                                       torch.logical_not(true_label)).sum().float().item()
                # TN = torch.logical_and(torch.logical_not(true_pred),
                #                        torch.logical_not(true_label)).sum().float().item()
                # print(TP+FP+FN+TN)
                TP_per_class.append(TP)
                FP_per_class.append(FP)
                FN_per_class.append(FN)
    return TP_per_class,FP_per_class,FN_per_class


def mIoU(pred_mask, mask, n_class=34):
    pred_mask = F.softmax(pred_mask)
    pred_mask = torch.argmax(pred_mask, dim=1)

    iou_per_class = []
    for c in range(0, n_class + 1):
        true_pred = pred_mask == c
        true_label = mask == c
        if true_label.long().sum().item() == 0:  # no exist label in this loop
            iou_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_pred, true_label).sum().float().item()
            union = torch.logical_or(true_pred, true_label).sum().float().item()
            iou = intersect / union
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class)


def show_result(train_loss, train_mIoU, train_pIoU, val_loss, val_mIoU, val_pIoU):
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(np.arange(len(train_loss)) + 1, train_loss)
    ax1.set_title('train loss')

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(np.arange(len(train_mIoU)) + 1, train_mIoU)
    ax2.set_title('train mIoU')

    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(np.arange(len(train_pIoU)) + 1, train_pIoU)
    ax3.set_title('train pIoU')

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(np.arange(len(val_loss)) + 1, val_loss)
    ax4.set_title('val loss')

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(np.arange(len(val_mIoU)) + 1, val_mIoU)
    ax5.set_title('val mIoU')

    ax6 = plt.subplot(2, 3, 6)
    ax6.bar(np.arange(len(val_pIoU)) + 1, val_pIoU)
    ax6.set_title('val.pIoU')
    plt.show()


def fit(model, n_class, epochs, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)
    torch.cuda.empty_cache()
    train_loss_list = []
    train_mIoU_list = []
    val_loss_list = []
    val_mIoU_list = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        train_mIoU = 0
        train_TP = np.zeros(n_class + 1)
        train_FP = np.zeros(n_class + 1)
        train_FN = np.zeros(n_class + 1)
        # train loop
        for i, data in enumerate(train_loader):
            image, mask = data[0].to(device), data[1].to(device)

            # forward
            out = model(image)
            loss = criterion(out, mask)

            # evaluation
            running_loss += loss.item()
            train_mIoU += mIoU(out, mask, n_class)
            TP,FP,FN= pIoU(out, mask, n_class)
            train_TP+=TP
            train_FP+=FP
            train_FN+=FN

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('epoch:', e, 'runningloss', running_loss)
        train_loss_list.append(running_loss)
        train_mIoU_list.append(train_mIoU)
        train_pIoU = train_TP/(train_TP+train_FP+train_FN)

        # validation
        model.eval()
        val_loss = 0
        val_mIoU = 0
        val_TP = np.zeros(n_class + 1)
        val_FP = np.zeros(n_class + 1)
        val_FN = np.zeros(n_class + 1)
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                image, mask = data[0].to(device), data[1].to(device)
                out = model(image)
                val_loss += criterion(out, mask).item()
                val_mIoU += mIoU(out, mask, n_class)
                TP, FP, FN = pIoU(out, mask, n_class)
                val_TP += TP
                val_FP += FP
                val_FN += FN

            print('epoch:', e, 'val loss', val_loss)
            val_loss_list.append(val_loss)
            print('epoch:', e, 'val mIoU', val_mIoU)
            val_mIoU_list.append(val_mIoU)
            val_pIoU=val_TP/(val_TP+val_FP+val_FN)

    show_result(train_loss_list, train_mIoU_list, train_pIoU, val_loss_list, val_mIoU_list, val_pIoU)
    return model


if __name__ == '__main__':
    # split dataset into test, val, train
    # split_dataset('./data_semantics/training/image_2/')
    # split_dataset('./data_semantics/training/semantic/',mode='mask')

    # hyper parameter
    max_lr = 1e-3
    epoch = 30
    weight_decay = 1e-4
    batchsize = 10
    n_class = 34

    # initial net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fcn16 = FCN16.Fcn16()
    fcn32 = FCN32.Fcn32()

    # load data
    train_files = [os.path.splitext(f)[0] for f in os.listdir('./data/image/train/')]
    train_set = K.KittiDataset('./data/image/train/', './data/mask/train/',
                             train_files, resize=(1242,375))
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_files = [os.path.splitext(f)[0] for f in os.listdir('./data/image/val/')]
    val_set = K.KittiDataset('./data/image/val/', './data/mask/val/',
                           val_files, resize=(1242,375))
    val_loader = DataLoader(val_set, batch_size=batchsize, shuffle=True)

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer16 = torch.optim.AdamW(fcn16.parameters(), lr=max_lr, weight_decay=weight_decay)
    optimizer32 = torch.optim.AdamW(fcn32.parameters(), lr=max_lr, weight_decay=weight_decay)
    # train net and show evaluation
    net_32 = fit(fcn32, n_class, epoch, train_loader, val_loader, criterion, optimizer32, device)
    PATH32 = './fcn32.pth'
    torch.save(net_32.state_dict(), PATH32)
    net_16 = fit(fcn16, n_class, epoch, train_loader, val_loader, criterion, optimizer16, device)
    PATH16 = './fcn16.pth'
    torch.save(net_16.state_dict(), PATH16)
