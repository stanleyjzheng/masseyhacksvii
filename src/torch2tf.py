import torch
from torch.autograd import Variable
import onnx
import tensorflow as tf
import onnx_tf

def cache_model(enet_type, device, model_file):
    model = EffnetMeta(enet_type, n_meta_features=0, out_dim=9)
    model = model.to(device)
    state_dict = torch.load(model_file)
    state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

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

if __name__ == '__main__':
    device = torch.device('cuda')
    model = cache_model('efficientnet-b7', device, '../input/model/9c_b7ns_1e_640_ext_15ep_best_fold0.pth')

    dummy_input = Variable(torch.randn(1, 1, 28, 28))
    torch.onnx.export(model, dummy_input, "../model/9c_b7ns_1e_640_ext_15ep_best_fold0.onnx")

    onnx_model = onnx.load("../model/9c_b7ns_1e_640_ext_15ep_best_fold0.onnx")
    tf_model = onnx_tf.prepare(onnx_model)

    tf_model.save_weights('../model/9c_b7ns_1e_640_ext_15ep_best_fold0.h5')