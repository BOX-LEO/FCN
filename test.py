import FCN32
import FCN16
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import labels as L
import KittiDataset as K


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


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
    return TP_per_class, FP_per_class, FN_per_class


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


def test(model, data_loader, criterion, op_device):
    model.to(op_device)
    model.eval()
    torch.cuda.empty_cache()
    test_TP = np.zeros(n_class + 1)
    test_FP = np.zeros(n_class + 1)
    test_FN = np.zeros(n_class + 1)
    with torch.no_grad():
        test_loss = 0
        test_mIoU = 0
        for i, data in enumerate(data_loader):
            image, mask = data[0].to(op_device), data[1].to(op_device)
            # print(image.size())
            out = model(image)

            test_loss += criterion(out, mask).item()
            test_mIoU += mIoU(out, mask, n_class)
            TP, FP, FN = pIoU(out, mask, n_class)
            test_TP += TP
            test_FP += FP
            test_FN += FN
            out_mask = F.softmax(out)
            out_mask = torch.argmax(out_mask, dim=1)
            mask2image(torchvision.utils.make_grid(out_mask[0, :, :].cpu()))
    print('test loss=', test_loss)
    print('test mIoU', test_mIoU)
    test_pIoU = test_TP / (test_TP + test_FP + test_FN)
    plt.bar(np.arange(len(test_pIoU)) + 1, test_pIoU)
    plt.title('test pIoU')
    plt.show()


def imshow(img):
    try:
        npimg = img.numpy()
    except:
        npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def mask2image(mask):
    _,H,W = mask.size()
    img = np.zeros((3,H,W),dtype=np.int)
    for h in range(H):
        for w in range(W):
            id = int(mask[0,h,w])
            color = L.id2label[id].color
            img[:,h,w]=color
    imshow(img)


if __name__ == '__main__':
    batchsize = 15
    n_class = 34

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fcn16 = FCN16.Fcn16()
    fcn32 = FCN32.Fcn32()
    fcn16.load_state_dict(torch.load('./fcn16.pth'))
    fcn32.load_state_dict(torch.load('./fcn32.pth'))

    # load data
    test_files = [os.path.splitext(f)[0] for f in os.listdir('./data/image/test/')]
    test_set = K.KittiDataset('./data/image/test/', './data/mask/test/',
                            test_files, resize=(1242, 375))
    test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False)

    # set criterion
    criterion = nn.CrossEntropyLoss()

    # test
    test(fcn32, test_loader, criterion, device)
    test(fcn16, test_loader, criterion, device)
