from utils import seed_everything, process_train_data, process_external_data, process_metadata, valid_one_epoch, \
    prepare_train_dataloader, EffnetMeta
from warmup_scheduler import GradualWarmupSchedulerV2
import torch
import time
import pandas as pd
import json
import albumentations as albu
from torch.cuda.amp import autocast, GradScaler  # for training only, need pytorch 1.7 or later
import os
from albumentations.pytorch import ToTensorV2


def train_transforms():
    return albu.Compose([
        albu.Transpose(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightness(limit=0.2, p=0.75),
        albu.RandomContrast(limit=0.2, p=0.75),
        albu.OneOf([
            albu.MotionBlur(blur_limit=5),
            albu.MedianBlur(blur_limit=5),
            albu.GaussianBlur(blur_limit=5),
            albu.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albu.OneOf([
            albu.OpticalDistortion(distort_limit=1.0),
            albu.GridDistortion(num_steps=5, distort_limit=1.),
            albu.ElasticTransform(alpha=3),
        ], p=0.7),

        albu.CLAHE(clip_limit=4.0, p=0.7),
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albu.Resize(CFG['img_size'], CFG['img_size']),
        albu.Cutout(max_h_size=int(CFG['img_size'] * 0.375), max_w_size=int(CFG['img_size'] * 0.375), num_holes=1,
                    p=0.7),
        ToTensorV2()
    ])


def loss_count(y_true_img, y_pred_img, device):
    bce_func = torch.nn.CrossEntropyLoss().to(device)
    image_loss = bce_func(y_pred_img, y_true_img)
    counts = y_true_img.size()[0]
    return image_loss, counts


def train_one_epoch(epoch, model, device, scaler, optimizer, train_loader):
    model.train()

    t = time.time()
    loss_sum = 0
    loss_w_sum = 0

    for step, (imgs, image_labels) in enumerate(train_loader):
        if CFG['metadata']:
            imgs, meta = imgs
            meta = meta.to(device).float()
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()

            with autocast():
                image_preds = model(imgs, meta)
        else:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()

            with autocast():
                image_preds = model(imgs)

        image_loss, counts = loss_count(image_labels, image_preds, device)

        loss = image_loss / counts
        scaler.scale(loss).backward()

        loss_sum += image_loss.detach().item()
        loss_w_sum += counts

        # gradient accumulation
        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            print(
                f'epoch {epoch} train step {step + 1}/{len(train_loader)}, ' +
                f'loss: {loss_sum / loss_w_sum:.5f}, ' +
                f'time: {(time.time() - t):.3f}', end='\r' if (step + 1) != len(train_loader) else '\n'
            )
    return loss_sum / loss_w_sum  # train loss


if __name__ == '__main__':
    with open('config.json') as json_file:
        CFG = json.load(json_file)

    seed_everything(42)

    train_df = pd.read_csv(CFG['train_path'])
    test_df = pd.read_csv(CFG['test_path'])
    external_df = pd.read_csv(CFG['ext_data_path'])

    train_df = process_train_data(train_df)
    external_df = process_external_data(external_df)
    test_df['filepath'] = test_df['image_name'].apply(lambda x: os.path.join(CFG['test_img_path'], f'{x}.jpg'))

    train_df = pd.concat([train_df, external_df]).reset_index(drop=True)
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(train_df.diagnosis.unique()))}
    train_df['target'] = train_df['diagnosis'].map(diagnosis2idx)
    mel_idx = diagnosis2idx['melanoma']

    if CFG['metadata']:
        meta_features, n_meta_features, train_df, test_df = process_metadata(train_df, test_df)
    else:
        n_meta_features = 0
        meta_features = []

    for fold in range(4):
        print('Fold:', fold + 1)
        train_loader, val_loader = prepare_train_dataloader(fold, train_df, meta_features, train_transforms)

        device = torch.device(CFG['device'])
        model = EffnetMeta(n_meta_features=n_meta_features, out_dim=9).to(device)
        scaler = GradScaler()
        num_epochs = CFG['freeze_epo'] + CFG['warmup_epo'] + CFG['cosine_epo']

        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['init_lr'])

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG['cosine_epo'])
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=CFG['warmup_epo'],
                                                    after_scheduler=scheduler_cosine)

        for epoch in range(num_epochs):
            train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)

            with torch.no_grad():
                valid_one_epoch(epoch, model, device, val_loader, mel_idx, loss_fn=loss_count)

        torch.save(model.state_dict(), '{}/model_fold_{}_{}'.format(CFG['save_path'], fold, CFG['tag']))
