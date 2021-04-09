import torch.nn as nn
import torch
from clipNet.mfb import CoAtt
import clip

class Net(nn.Module):
    def __init__(self,  args, answer_size):
        super(Net, self).__init__()
        self.args = args

        self.model, self.preprocess = clip.load("ViT-B/32")


        self.backbone = CoAtt(args)

        if args.HIGH_ORDER:  # MFH 默认FALSE
            self.proj = nn.Linear(2 * args.MFB_O, answer_size) # MFB_O=1000
        else:  # MFB
            self.proj = nn.Linear(args.MFB_O, answer_size)

    def forward(self, img, text):


        image_features = self.model.encode_image(img)

        text_feature = self.model.encode_text(text)

        z = self.backbone(image_features.float(), text_feature.float())  # [batch, 1000]

        proj_feat = self.proj(z)  # FC:1000*3129

        return proj_feat  # [batch, 3129], 3129刚好就是answers的数量, 用于计算cross_entropy loss