'''
Based on https://github.com/microsoft/vscode/issues/125993 
train attribute_aware network with known dataset
python -u train_ATRB.py 20 full fitzpatrick ATRB
python -u train_ATRB.py 15 full ddi ATRB
'''
from __future__ import print_function, division
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import cv2
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter
# get model
from Models.models_losses import Network

np.random.seed(42)
warnings.filterwarnings("ignore")


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    train_step = 0 # for tensorboard
    leading_epoch = 0  # record best model epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_balanced_acc_sum = 0.0
            # running_total = 0
            print(phase)
            # Iterate over data.
            for n_iter, batch in enumerate(dataloaders[phase]):
                inputs = batch["image"].to(device)
                labels = batch[label]
                # change skin type to one hot vector note!!!!! attributes 1-6
                attributes = batch['fitzpatrick']-1
                input_att = torch.nn.functional.one_hot(attributes, 6).float().to(device)
                labels = torch.from_numpy(np.asarray(labels)).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.float()  # ADDED AS A FIX
                    outputs = model(inputs, input_att)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                # tensorboard
                if phase == 'train':
                    writer.add_scalar('Loss/'+phase, loss.item(), train_step)
                    writer.add_scalar('Accuracy/'+phase, (torch.sum(preds == labels.data)).item()/inputs.size(0), train_step)
                    writer.add_scalar('Balanced-Accuracy/'+phase, balanced_accuracy_score(labels.data.cpu(), preds.cpu()), train_step)
                    train_step += 1
                # -------------------------
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_balanced_acc_sum += balanced_accuracy_score(labels.data.cpu(), preds.cpu())*inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects,
                                           dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f} Balanced-Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_balanced_acc))
            # tensorboard 
            writer.add_scalar('lr/'+phase, scheduler.get_last_lr()[0], epoch)
            if phase == 'val':
                writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
                writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch)
                writer.add_scalar('Balanced-Accuracy/'+phase, epoch_balanced_acc, epoch)
            # ---------------------    
            training_results.append([phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc])
            if epoch > 0:
                if phase == 'val' and epoch_acc > best_acc:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    leading_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                #### use balanced acc
                # if phase == 'val' and epoch_balanced_acc > best_acc:
                #     print("New leading balanced accuracy: {}".format(epoch_balanced_acc))
                #     best_acc = epoch_balanced_acc
                #     leading_epoch = epoch
                #     best_model_wts = copy.deepcopy(model.state_dict())     
            elif phase == 'val':
                best_acc = epoch_acc
                # best_loss = epoch_loss
                # use balanced acc
                # best_acc = epoch_balanced_acc
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best model epoch:', leading_epoch)
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy", "balanced-accuracy"]
    return model, training_results


class SkinDataset():
    def __init__(self, dataset_name, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.dataset_name == 'ddi':
            img_name = os.path.join(self.root_dir,
                                str(self.df.loc[self.df.index[idx], 'hasher']))
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_name = os.path.join(self.root_dir,
                                str(self.df.loc[self.df.index[idx], 'hasher']))+'.jpg'
            image = io.imread(img_name)

        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        high = self.df.loc[self.df.index[idx], 'high']
        # mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']

        if self.dataset_name == 'fitzpatrick':
            mid = self.df.loc[self.df.index[idx], 'mid'] 
        else:
            mid = 0

        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'high': high,
                    'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick': fitzpatrick
                }
        return sample


def custom_load(
        batch_size=128,
        num_workers=10,
        train_dir='',
        val_dir='',
        label = 'low',
        dataset_name = 'fitzpatrick',
        image_dir='/bigdata/siyiplace/data/skin_lesion/fitzpatrick17k/data/finalfitz17k/',
        ):
    if dataset_name == 'ddi':
        image_dir = '/bigdata/siyiplace/data/skin_lesion/ddidiversedermatologyimages/'

    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    class_sample_count = np.array(train[label].value_counts().sort_index())
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train[label]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True)
    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}
    transformed_train = SkinDataset(
        dataset_name = dataset_name,
        csv_file=train_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])
        )
    transformed_test = SkinDataset(
        dataset_name = dataset_name,
        csv_file=val_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # shuffle=True,
            num_workers=num_workers),
        "val": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
        }
    return dataloaders, dataset_sizes




if __name__ == '__main__':
    # In the custom_load() function, make sure to specify the path to the images
    print("\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n")
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    dataset_name = sys.argv[3]
    model_name = sys.argv[4]
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_mode == "dev":
        if dataset_name == 'ddi':
            df = pd.read_csv('ddi_metadata_code.csv').sample(300)
        else:
            df = pd.read_csv("fitzpatrick17k_known_code.csv").sample(1000)
    else:
        if dataset_name == 'ddi':
            df = pd.read_csv('ddi_metadata_code.csv')
        else:
            df = pd.read_csv("fitzpatrick17k_known_code.csv")
    print(df['fitzpatrick'].value_counts())  # count num of samples in each fitzpatrick type
    print("Rows: {}".format(df.shape[0]))
    # df["low"] = df['label'].astype('category').cat.codes


    for holdout_set in ["a56"]: # ["expert_select","random_holdout", "a12", "a34","a56", "dermaamin","br"]:
        if holdout_set == "expert_select":
            df2 = df
            train = df2[df2.qc.isnull()]
            test = df2[df2.qc=="1 Diagnostic"]
        elif holdout_set == "random_holdout":
            if dataset_name == 'ddi':
                train, test, y_train, y_test = train_test_split(
                                                    df,
                                                    df['high'],
                                                    test_size=0.2,
                                                    random_state=64,
                )
            else:
                train, test, y_train, y_test = train_test_split(
                                                    df,
                                                    df['low'],
                                                    test_size=0.2,
                                                    random_state=64,
                                                    stratify=df['low']) # 
        elif holdout_set == "dermaamin": # train with b
            # only choose those skin conditions in both dermaamin and non dermaamin
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df = df[df.label.isin(combo)]
            df["low"] = df['label'].astype('category').cat.codes
            train = df[df.image_path.str.contains("dermaamin") == False]
            test = df[df.image_path.str.contains("dermaamin")]
        elif holdout_set == "br": # train with a
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df = df[df.label.isin(combo)]
            df["low"] = df['label'].astype('category').cat.codes
            train = df[df.image_path.str.contains("dermaamin")]
            test = df[df.image_path.str.contains("dermaamin") == False]
            print(train.label.nunique())
            print(test.label.nunique())
        elif holdout_set == "a12":
            train = df[(df.fitzpatrick==1)|(df.fitzpatrick==2)]
            test = df[(df.fitzpatrick!=1)&(df.fitzpatrick!=2)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print(combo)
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a34":
            train = df[(df.fitzpatrick==3)|(df.fitzpatrick==4)]
            test = df[(df.fitzpatrick!=3)&(df.fitzpatrick!=4)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a56":
            train = df[(df.fitzpatrick==5)|(df.fitzpatrick==6)]
            test = df[(df.fitzpatrick!=5)&(df.fitzpatrick!=6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        print(train.shape)
        print(test.shape)
        train_path = "temp_train_{}.csv".format(model_name)
        test_path = "temp_test_{}.csv".format(model_name)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print("Training Shape: {}, Test Shape: {} \n".format(
        train.shape,
        test.shape)
        )
        for indexer, label in enumerate(["high"]):
            # tensorboard
            writer = SummaryWriter(comment="logs_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            print(label)
            weights = np.array(max(train[label].value_counts())/train[label].value_counts().sort_index())
            label_codes = sorted(list(train[label].unique()))
            dataloaders, dataset_sizes = custom_load(
                64,
                10,
                "{}".format(train_path),
                "{}".format(test_path),
                label = label,
                dataset_name = dataset_name)


            model_ft = Network('attribute_aware', [len(label_codes), 6], pretrained=True)

            total_params = sum(p.numel() for p in model_ft.parameters())
            print('{} total parameters'.format(total_params))
            total_trainable_params = sum(
                p.numel() for p in model_ft.parameters() if p.requires_grad)
            print('{} total trainable parameters'.format(total_trainable_params))
            model_ft = model_ft.to(device)
            model_ft = nn.DataParallel(model_ft)
            class_weights = torch.FloatTensor(weights).cuda()
            # criterion = nn.NLLLoss()
            criterion = nn.CrossEntropyLoss()
            optimizer_ft = optim.Adam(model_ft.parameters(), 0.0001)
            # exp_lr_scheduler = lr_scheduler.StepLR(
            #     optimizer_ft,
            #     step_size=7,
            #     gamma=0.1)
            exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft,
            step_size=2,
            gamma=0.9)

            print("\nTraining classifier for {}........ \n".format(label))
            print("....... processing ........ \n")
            model_ft, training_results = train_model(
                label,
                dataloaders, device,
                dataset_sizes, model_ft,
                criterion, optimizer_ft,
                exp_lr_scheduler, n_epochs)
            print("Training Complete")

            torch.save(model_ft.state_dict(), "model_path_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            torch.save(model_ft, "model_path_{}_{}_{}_{}.pt".format(model_name, n_epochs, label, holdout_set))
            print("gold")
            training_results.to_csv("training_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set))
            model = model_ft.eval()
            loader = dataloaders["val"]
            prediction_list = []
            fitzpatrick_list = []
            hasher_list = []
            labels_list = []
            p_list = []
            topk_p = []
            topk_n = []
            d1 = []
            d2 = []
            d3 = []
            p1 = []
            p2 = []
            p3 = []
            with torch.no_grad():
                running_corrects = 0
                running_balanced_acc_sum  = 0
                total = 0
                for i, batch in enumerate(dataloaders['val']):
                    inputs = batch["image"].to(device)
                    classes = batch[label].to(device)
                    fitzpatrick = batch["fitzpatrick"]  # skin type
                    hasher = batch["hasher"]
                    # attribute note!!! subtract 1
                    input_att = torch.nn.functional.one_hot(fitzpatrick-1, 6).float().to(device)
                    outputs = model(inputs.float(), input_att)  # (batchsize, classes num)
                    probability = torch.nn.functional.softmax(outputs, dim=1)
                    ppp, preds = torch.topk(probability, 1) #topk values, topk indices
                    if label == "low":
                        _, preds5 = torch.topk(probability, 3)  # topk values, topk indices
                        # topk_p.append(np.exp(_.cpu()).tolist())
                        topk_p.append((_.cpu()).tolist())
                        topk_n.append(preds5.cpu().tolist())
                    running_corrects += torch.sum(preds.reshape(-1) == classes.data)
                    running_balanced_acc_sum += balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu()) * inputs.shape[0]
                    p_list.append(ppp.cpu().tolist())
                    prediction_list.append(preds.cpu().tolist())
                    labels_list.append(classes.tolist())
                    fitzpatrick_list.append(fitzpatrick.tolist())
                    hasher_list.append(hasher)
                    total += inputs.shape[0]
                acc = float(running_corrects)/float(dataset_sizes['val'])
                balanced_acc = float(running_balanced_acc_sum)/float(dataset_sizes['val'])
            if label == "low":
                for j in topk_n: # each sample
                    for i in j:  # in k
                        d1.append(i[0])
                        d2.append(i[1])
                        d3.append(i[2])
                for j in topk_p:
                    for i in j:
                        # print(i)
                        p1.append(i[0])
                        p2.append(i[1])
                        p3.append(i[2])
                df_x=pd.DataFrame({
                                    "hasher": flatten(hasher_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list),
                                    "d1": d1,
                                    "d2": d2,
                                    "d3": d3,
                                    "p1": p1,
                                    "p2": p2,
                                    "p3": p3})
            else:
                # print(len(flatten(hasher_list)))
                # print(len(flatten(labels_list)))
                # print(len(flatten(fitzpatrick_list)))
                # print(len(flatten(p_list)))
                # print(len(flatten(prediction_list)))
                df_x=pd.DataFrame({
                                    "hasher": flatten(hasher_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list)})
            df_x.to_csv("results_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set),
                            index=False)
            print("\n Accuracy: {}  Balanced Accuracy: {} \n".format(acc, balanced_acc))
        print("done")
        # writer.close()
