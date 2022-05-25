import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import random
import numpy as np
import argparse
import os
from kappa import kappa

my_whole_seed = 111
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False

import shutil
import time
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
#from torch.utils.tensorboard import SummaryWriter


best_acc1 = 0
best_auc = 0
best_kap = 0
best_accdr = 0
minimum_loss = 1.0
count = 0

def main():
    global best_acc1
    global best_auc
    global best_kap
    global minimum_loss
    global count
    global best_accdr
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    # load model
    from models.resnet50 import resnet50
    model = resnet50(num_classes=5, origin=True)
    # model = resnet50(num_classes=5, simple= True, test=True, crossCBAM=True)

    print ("==> Load pretrained model")
    model_dict = model.state_dict()
    pretrain_path = "./resnet50-19c8e357.pth"
    # pretrain_path = "./resnet18-5c106cde.pth"
    pretrained_dict = torch.load(pretrain_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict.pop('fc.weight', None)
    pretrained_dict.pop('fc.bias', None)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(device)

    # print(model)
    # exit(0)


    # data argument and load data
    transform = transforms.Compose(
        [
            # transforms.Resize(512),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            # transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            # transforms.Resize(512),
            transforms.CenterCrop(224),
            # transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # more argument
    # transform = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomVerticalFlip(),
    #             # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    #             transforms.RandomRotation([-180, 180]),
    #             transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
    #             transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #         ]
    #     )
    # transform_test = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #         ]
    #     )


    from dataset import dataset
    train_dataset = dataset(csv_path='./dataset/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv',
                         root='./dataset/regular_fundus_images/regular-fundus-training/Images', transform=transform)
    val_dataset = dataset(
        csv_path='./dataset/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv',
        root='./dataset/regular_fundus_images/regular-fundus-validation/Images', transform=transform_test)

    batch_size = 24
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)


    # define loss function (criterion) and optimizer

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.SmoothL1Loss()
    # criterion3 = MaxLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 原文使用cosine annealing for each batch
    # from lr_scheduler import LRScheduler
    # lr_scheduler = LRScheduler(optimizer, len(train_loader))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    n_epochs = 50
    for epoch in range(n_epochs):

        print("-" * 10)
        print("Epoch {}/{}".format(epoch + 1, n_epochs))

        is_best = False
        is_best_kap = False
        is_best_acc = False

        # train for one epoch
        loss_train, acc_train, kappa_train = train(train_loader, model, criterion1, criterion2, lr_scheduler, epoch, optimizer,device)
        # writer.add_scalar('Train loss', loss_train, epoch)

        # evaluate on validation set
        acc, kappa_val, precision_dr, recall_dr, f1score_dr = validate(val_loader, model, device)

        is_best = acc >= best_acc1
        best_acc1 = max(acc, best_acc1)

        is_best_kap = kappa_val >= best_kap
        best_kap = max(kappa_val, best_kap)

        print("Train Loss:{:.4f}, Train Acc:{:.4f}%, Val Acc:{:.4f}%".format(loss_train, 100*acc_train,100*acc))
        print("Train Kappa:{:.4f}, Val Kappa:{:.4f}".format(kappa_train,kappa_val))
        print("Best Acc:{:.4f}%, Best Kappa:{:.4f}".format(100 * best_acc1,best_kap))


def train(train_loader, model, criterion, criterion2, lr_scheduler, epoch, optimizer,device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    all_target = []
    all_output = []

    for i, (input, target, patient_level) in enumerate(train_loader):

        # data = [item.to(device) for item in data]
        input = [item.to(device) for item in input]
        target = [item.to(device) for item in target]
        patient_level = patient_level.to(device)


        ### multi input
        # output = model(input[0],input[1])
        # _, pred1 = torch.max(output[0].data, 1)
        # _, pred2 = torch.max(output[1].data, 1)
        # 两个图分别的loss
        # loss1 = criterion(output[0], target[0])
        # loss2 = criterion(output[1], target[1])
        # # 两个图smoothL1
        # loss3 = criterion2(output[0], output[1])
        # # 交叉attention之后的最终类别loss
        # # loss4 = criterion(output[2], patient_level)

        ## single input
        output1 = model(input[0])
        output2 = model(input[0])
        _, pred1 = torch.max(output1.data, 1)
        _, pred2 = torch.max(output2.data, 1)

        loss1 = criterion(output1, target[0])
        loss2 = criterion(output2, target[1])
        # 两个图smoothL1
        loss3 = criterion2(output1, output2)

        # print(pred1,pred2)
        output_patient = torch.max(pred1,pred2)

        all_output.append(pred1.cpu().data.numpy())
        all_output.append(pred2.cpu().data.numpy())
        # all_output.append(output_patient.cpu().data.numpy())
        all_target.append(target[0].cpu().data.numpy())
        all_target.append(target[1].cpu().data.numpy())

        # loss = loss1 + loss2 + loss3 + loss4
        # loss = loss1 + loss2 + loss3
        loss = loss1 + loss2

        losses.update(loss.item(), len(patient_level))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('train epoch: {} \tloss: {:.6f}\t'.format(epoch + 1, loss.item()))

    all_target = [item for sublist in all_target for item in sublist]
    all_output = [item for sublist in all_output for item in sublist]
    # acc_train = accuracy_score(all_target, np.argmax(all_output, axis=1))
    acc_train = accuracy_score(all_target, all_output)
    kappa_train = quadratic_kappa(all_target, all_output)

    return losses.avg, acc_train, kappa_train

from sklearn.metrics import confusion_matrix
import scipy.misc

def validate(val_loader, model, device):
    # switch to evaluate mode
    model.eval()

    all_target = []
    all_output = []
    with torch.no_grad():
        for i, (input, target, patient_level) in enumerate(val_loader):
            input = [item.to(device) for item in input]
            target = [item.to(device) for item in target]
            patient_level = patient_level.to(device)

            ### multi input
            # output = model(input[0],input[1])
            # _, pred1 = torch.max(output[0].data, 1)
            # _, pred2 = torch.max(output[1].data, 1)
            # 两个图分别的loss
            # loss1 = criterion(output[0], target[0])
            # loss2 = criterion(output[1], target[1])
            # # 两个图smoothL1
            # loss3 = criterion2(output[0], output[1])
            # # 交叉attention之后的最终类别loss
            # # loss4 = criterion(output[2], patient_level)

            ## single input
            output1 = model(input[0])
            output2 = model(input[0])
            _, pred1 = torch.max(output1.data, 1)
            _, pred2 = torch.max(output2.data, 1)

            # output_patient = torch.max(pred1, pred2)

            all_output.append(pred1.cpu().data.numpy())
            all_output.append(pred2.cpu().data.numpy())
            # all_output.append(output_patient.cpu().data.numpy())
            all_target.append(target[0].cpu().data.numpy())
            all_target.append(target[1].cpu().data.numpy())


    all_target = [item for sublist in all_target for item in sublist]
    all_output = [item for sublist in all_output for item in sublist]

    kappa_val = quadratic_kappa(all_target, all_output)

    # acc = accuracy_score(all_target, np.argmax(all_output, axis=1))
    # auc = multi_class_auc(all_target, np.argmax(all_output, axis=1), num_c=5)
    # precision_dr = precision_score(all_target, np.argmax(all_output, axis=1), average="macro",zero_division=0)
    # recall_dr = recall_score(all_target, np.argmax(all_output, axis=1), average="macro")
    # f1score_dr = f1_score(all_target, np.argmax(all_output, axis=1), average="macro")

    acc = accuracy_score(all_target, all_output)
    # auc = multi_class_auc(all_target, all_output, num_c=5)
    precision_dr = precision_score(all_target, all_output, average="macro",zero_division=0)
    recall_dr = recall_score(all_target, all_output, average="macro")
    f1score_dr = f1_score(all_target, all_output, average="macro")

    return acc, kappa_val, precision_dr, recall_dr, f1score_dr


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_dir= 'file'):

    root = save_dir + "/"
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(state, root+filename)


def save_result2txt(savedir, all_output_dme, all_output,all_target_dme,all_target):
    np.savetxt(savedir+"/output_dme.txt", all_output_dme, fmt='%.4f')
    np.savetxt(savedir+"/output_dr.txt", all_output, fmt='%.4f')
    np.savetxt(savedir+"/target_dme.txt", all_target_dme)
    np.savetxt(savedir+"/target_dr.txt", all_target)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def multi_class_auc(all_target, all_output, num_c = None):

    all_output = np.stack(all_output)
    all_target = label_binarize(all_target, classes=list(range(0, num_c)))
    auc_sum = []

    for num_class in range(0, num_c):
        try:
            auc = roc_auc_score(all_target[:, num_class], all_output[:, num_class])
            auc_sum.append(auc)
        except ValueError:
            pass

    auc = sum(auc_sum) / float(len(auc_sum))

    return auc

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels].cuda()


def save_result_txt(savedir, result):

    with open(savedir + '/result.txt', 'w') as f:
        for item in result:
            f.write("%.8f\n" % item)
        f.close()


def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
    of adoption rating."""
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum()
    O = O / O.sum()

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    return (1 - (num / den))

if __name__ == '__main__':
    main()
