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


def predict(image_path, age, site, gender):
    image_size = 9.783024603416145 # average since this is not relevant to our dataset

    gender_map = {'male': 1, 'female': 0}

    site_map = {
        'anterior torso': 0,
        'head/neck': 1,
        'lateral torso': 2,
        'lower extremity': 3,
        'oral': 4,
        'palms/soles': 5,
        'posterior torso': 6,
        'torso': 7,
        'upper extremity': 8,
        'none of the above': 9
    }

    gender = gender_map[gender]
    age /= 90

    site_oh = np.zeros(10)
    site_oh[site_map[site]] += 1

    metadata = [gender, age, 1, image_size] + site_oh.tolist()

    print(metadata)

    device = torch.device('cuda')

    image = cv2.imread(image_path)
    image = image[:, :, ::-1]
    res = transforms_val(640)(image=image)
    image = res['image'].astype(np.float32)
    image = image.transpose(2, 0, 1)

    image = torch.tensor(image).float().to(device)
    metadata = torch.tensor(metadata).float().to(device)

    model = cache_model('efficientnet-b7', device, '../input/model/9c_b7ns_1e_640_ext_15ep_best_fold0.pth')

    logits = model(image.unsqueeze(axis=0), metadata)
    probs = logits.softmax(1).detach().cpu().numpy()
    return probs


def process_preds(preds):
    classes = {
        0: 'Actinic Keratosis',
        1: 'Basal Cell Carcinoma',
        2: 'Benign Keratosis',
        3: 'Dermatofibroma',
        4: 'Squamous Cell Carcinoma',
        5: 'Vascular Lesion',
        6: 'Melanoma',
        7: 'Nevus',
        8: 'Unknown'
    }

    m = np.argmax(preds[0])
    out = classes[m]

    if preds[0, 6]>0.5:
        melanoma_risk = 'High'
    elif preds[0, 6]<0.5 and preds[0, 6] > 0.3:
        melanoma_risk = 'Medium'
    else:
        melanoma_risk = 'Low'

    return out, melanoma_risk


if __name__ == '__main__':
    print(process_preds(predict('../input/jpeg-melanoma-256x256/train/ISIC_8901784.jpg', 42, 'oral', 'female')))