import json
import os
import random
import time

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import geffnet
from albumentations.pytorch import ToTensorV2


with open('config.json') as json_file:
    CFG = json.load(json_file)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def val_transforms():
    return albu.Compose([
        albu.Resize(CFG["img_size"], CFG["img_size"], p=1.0),
        ToTensorV2()
    ], p=1.0)


def valid_one_epoch(epoch, model, device, val_loader, mel_idx, loss_fn=None):
    model.eval()

    t = time.time()
    loss_sum = 0
    acc_sum = 0
    loss_w_sum = 0
    auc_sum = 0
    logits = []
    probs = []
    g_truth = []

    for step, (imgs, labels) in enumerate(val_loader):
        with torch.no_grad():
            if CFG['metadata']:
                imgs, meta = imgs
                meta = meta.to(device).float()
                imgs = imgs.to(device).float()
                labels = labels.to(device).long()

                # could do flips here rather trivially, don't know if it will help
                image_preds = model(imgs, meta)

            logits.append(image_preds.detach().cpu())
            probs.append(image_preds.softmax(1).detach().cpu())
            g_truth.append(labels.detach().cpu())

        image_loss, counts = loss_fn(labels, image_preds, device)

        probs_np = torch.cat(probs).numpy()
        g_truth_np = torch.cat(g_truth).numpy()
        auc = roc_auc_score(np.array(g_truth_np == mel_idx).astype(int), probs_np[:, 0])

        loss_sum += image_loss.detach().item()

        loss_w_sum += counts
        auc_sum += auc

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            print(
                f'epoch {epoch} valid Step {step + 1}/{len(val_loader)}, ' +
                f'loss: {loss_sum / loss_w_sum:.3f}, ' +
                f"Accuracy: {acc_sum / loss_w_sum:.3f}, " +
                f"AUC: {auc_sum / loss_w_sum:.3f}, " +
                f'time: {(time.time() - t):.2f}', end='\r' if (step + 1) != len(val_loader) else '\n'
            )
    return loss_sum / loss_w_sum, auc_sum / loss_w_sum


def process_train_data(train_df):
    train_df = train_df[train_df['tfrecord'] != -1].reset_index(drop=True)

    # Use the stratified kfold tfrecord numbers as folds
    tfrecord2fold = {
        2: 0, 4: 0, 5: 0,
        1: 1, 10: 1, 13: 1,
        0: 2, 9: 2, 12: 2,
        3: 3, 8: 3, 11: 3,
        6: 4, 7: 4, 14: 4,
    }
    train_df['fold'] = train_df['tfrecord'].map(tfrecord2fold)
    train_df['is_ext'] = 0
    train_df['filepath'] = train_df['image_name'].apply(
        lambda x: os.path.join(CFG['train_img_path'], f'{x}.jpg'))

    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(
        lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))
    return train_df


def process_external_data(external_df):
    external_df = external_df[external_df['tfrecord'] >= 0].reset_index(drop=True)
    external_df['fold'] = external_df['tfrecord'] % 5
    external_df['is_ext'] = 1
    external_df['filepath'] = external_df['image_name'].apply(
        lambda x: os.path.join(CFG['ext_data_img_path'], f'{x}.jpg'))

    external_df['diagnosis'] = external_df['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    external_df['diagnosis'] = external_df['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    return external_df


def process_metadata(train_df, test_df):
    concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']],
                       ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)

    train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
    test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
    train_df['sex'] = train_df['sex'].fillna(-1)
    test_df['sex'] = test_df['sex'].fillna(-1)

    train_df['age_approx'] /= 90
    test_df['age_approx'] /= 90
    train_df['age_approx'] = train_df['age_approx'].fillna(0)
    test_df['age_approx'] = test_df['age_approx'].fillna(0)
    train_df['patient_id'] = train_df['patient_id'].fillna(0)

    train_df['n_images'] = train_df.patient_id.map(train_df.groupby(['patient_id']).image_name.count())
    test_df['n_images'] = test_df.patient_id.map(test_df.groupby(['patient_id']).image_name.count())
    train_df.loc[train_df['patient_id'] == -1, 'n_images'] = 1
    train_df['n_images'] = np.log1p(train_df['n_images'].values)
    test_df['n_images'] = np.log1p(test_df['n_images'].values)

    train_images = train_df['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(train_images):
        train_sizes[i] = os.path.getsize(img_path)
    train_df['image_size'] = np.log(train_sizes)
    test_images = test_df['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])

    for i, img_path in enumerate(test_images):
        test_sizes[i] = os.path.getsize(img_path)
    test_df['image_size'] = np.log(test_sizes)
    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in train_df.columns if
                                                                       col.startswith('site_')]
    n_meta_features = len(meta_features)

    return meta_features, n_meta_features, train_df, test_df


class MelanomaDataset(Dataset):
    def __init__(self, csv, split, mode, meta_features, transforms=None):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transforms = transforms
        self.meta_features = meta_features

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transforms is not None:
            res = self.transforms(image=image)
            image = res['image']/255.
        else:
            image = image.astype(np.float32)

        if CFG['metadata']:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


class EffnetMeta(torch.nn.Module):
    def __init__(self, out_dim, n_meta_features=0, load_pretrained=True):

        super(EffnetMeta, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(CFG['efbnet'].replace('-', '_'), pretrained=load_pretrained)
        self.dropout = torch.nn.Dropout(0.5)
        in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = torch.nn.Sequential(
                torch.nn.Linear(n_meta_features, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(512, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.SiLU(),
            )
            in_ch += 128
        self.myfc = torch.nn.Linear(in_ch, out_dim)
        self.enet.classifier = torch.nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
            x = self.myfc(self.dropout(x))
        return x


def prepare_train_dataloader(fold, train_df, meta_features, train_transforms):
    fold_train_df = train_df[train_df['fold'] != fold]
    fold_valid_df = train_df[train_df['fold'] == fold]

    train_ds = MelanomaDataset(fold_train_df, 'train', 'train', meta_features, transforms=train_transforms())
    val_ds = MelanomaDataset(fold_valid_df, 'train', 'val', meta_features, transforms=val_transforms())

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=CFG['pin_memory'],
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        drop_last=False,
        shuffle=False,
        pin_memory=CFG['pin_memory'],
    )
    return train_loader, val_loader
