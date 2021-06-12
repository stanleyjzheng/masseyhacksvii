import torch
import numpy as np
import cv2
import PIL.Image
import geffnet
import albumentations as albu


class EffnetMeta(torch.nn.Module):
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=False, enet_type='efficientnet-b7'):

        super(EffnetMeta, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=load_pretrained)
        self.dropout = torch.nn.Dropout(0.5)

        in_ch = self.enet.classifier.in_features
        self.myfc = torch.nn.Linear(in_ch, out_dim)
        self.enet.classifier = torch.nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        x = self.myfc(self.dropout(x))
        return x


def transforms_val(image_size):
    return albu.Compose([
        albu.Resize(image_size, image_size),
        albu.Normalize()
    ])


def cache_model(enet_type, device, model_file):
    model = EffnetMeta(enet_type, n_meta_features=0, out_dim=9)
    model = model.to(device)
    state_dict = torch.load(model_file)
    state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def predict(image_path, metadata):
    device = torch.device('cuda')

    image = cv2.imread(image_path)
    image = image[:, :, ::-1]
    res = transforms_val(640)(image=image)
    print(res)
    image = res['image'].astype(np.float32)
    image = image.transpose(2, 0, 1)

    image = torch.tensor(image).float().to(device)
    metadata = torch.tensor(metadata).float().to(device)

    model = cache_model('efficientnet-b7', device, '../input/model/9c_b7ns_1e_640_ext_15ep_best_fold0.pth')

    logits = model(image, metadata)
    probs = logits.softmax(1).detach().cpu()
    return probs

if __name__ == '__main__':
    predict('../input/jpeg-melanoma-256x256/train/ISIC_8901784.jpg', [1, 1, 1, 1, 1, 1])