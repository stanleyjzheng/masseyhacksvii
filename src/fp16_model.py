import geffnet
import torch


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


def fp16_model(enet_type, model_file):
    device = torch.device('cuda')
    model = EffnetMeta(enet_type, n_meta_features=0, out_dim=9)
    model = model.to(device)
    state_dict = torch.load(model_file)
    state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    torch.save(model.state_dict(), 'fp16_'+model_file)
    return model

if __name__ == '__main__':
    fp16_model('efficientnet-b7', '../input/model/9c_b7ns_1e_640_ext_15ep_best_fold0.pth')
