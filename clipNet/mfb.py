# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from clipNet.fc import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------


class MFB(nn.Module):
    def __init__(self, __C, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.__C = __C
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, __C.MFB_K * __C.MFB_O)
        self.proj_q = nn.Linear(1024, __C.MFB_K * __C.MFB_O)
        self.dropout = nn.Dropout(__C.drop_rate)
        self.pool = nn.AvgPool1d(__C.MFB_K, stride=__C.MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> [batch, 100, 2048]    C = 1 or 100
            ques_feat.size() -> [batch, 1, 2048]
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)  # FC:2048*5000   [batch, 100, 5000]
        ques_feat = self.proj_q(ques_feat)  # FC: 2048*5000   [batch, 1, 5000]

        exp_out = img_feat * ques_feat    # [batch, 11, 5000]
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)      # [batch, 11, 5000]
        z = self.pool(exp_out) * self.__C.MFB_K   # MFB_K=5     [batch, 100, 1000]
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))  # 下面三行是不是就是power norm
        z = F.normalize(z.view(batch_size, -1))         # [batch, 100000]
        z = z.view(batch_size, -1, self.__C.MFB_O)      # [batch, 100, 1000]      MFB_O = 1000
        return z, exp_out # z = [batch, 100, 1000]  exp_out = [batch, 11, 5000]


class QAtt(nn.Module):
    def __init__(self, __C):
        super(QAtt, self).__init__()
        self.__C = __C
        self.mlp = MLP(
            in_size = 512,
            mid_size = __C.hidden_size,
            out_size = __C.q_glimse,
            dropout_r = __C.drop_rate,
            use_relu = True
        )

    def forward(self, ques_feat): # ques_feat: [batch, seq, 1024]
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # [batch, seq, 2] 两层FC层, 1024*512, 512*2
        qatt_maps = F.softmax(qatt_maps, dim=1)         # [batch, seq, q_glimse]

        qatt_feat_list = []
        for i in range(self.__C.q_glimse):
            mask = qatt_maps[:, i:i + 1]             # [batch, seq, 1]   [b, 1]
            mask = mask * ques_feat                     # [batch, seq, 1] * [batch, seq, 1024] = [batch, seq, 1024]
            # mask = torch.sum(mask, dim=1)               # [batch, 1024]
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # [batch, 2048]

        return qatt_feat                                # [batch, 2048]


class IAtt(nn.Module):
    def __init__(self, __C, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(__C.drop_rate)
        self.mfb = MFB(__C, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=__C.MFB_O,
            mid_size=__C.hidden_size,
            out_size=__C.i_glimse,
            dropout_r=__C.drop_rate,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat):
        '''
            img_feats.size() -> [batch, 100, 2048]
            ques_att_feat.size() -> [batch, 2048]
            iatt_feat.size() -> []
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)      # [batch, 1, 2048]
        img_feat = img_feat.unsqueeze(1)
        img_feat = self.dropout(img_feat)               # [batch, 100, 2048]
        z, _ = self.mfb(img_feat, ques_att_feat)        #  z = [batch, 100, 1000]  _ = [batch, 11, 5000]

        iatt_maps = self.mlp(z)                         # FC: 1000*512, 512*2  [batch, 100, 2]
        iatt_maps = F.softmax(iatt_maps, dim=1)         # [batch, 100, 2]
        iatt_maps = iatt_maps.squeeze(1)

        iatt_feat_list = []
        for i in range(self.__C.i_glimse):
            mask = iatt_maps[:, i:i + 1]             # [batch, 100, 1]
            mask = mask * img_feat                      # [batch, 100, 2048]
            mask = torch.sum(mask, dim=1)               # [batch, 2048]
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # [batch, 2048*2]

        return iatt_feat                                # [batch, 2048*2]


class CoAtt(nn.Module):
    def __init__(self, __C):
        super(CoAtt, self).__init__()
        self.__C = __C

        img_feat_size = 512 # int: 2048
        img_att_feat_size = img_feat_size * __C.i_glimse # int: 4096
        ques_att_feat_size = __C.lstm_out_size * __C.q_glimse # int: 2048

        self.q_att = QAtt(__C)
        self.i_att = IAtt(__C, img_feat_size, ques_att_feat_size)

        if self.__C.HIGH_ORDER:  # MFH
            self.mfh1 = MFB(__C, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(__C, img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB
            self.mfb = MFB(__C, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)               # [batch, 1024]
        fuse_feat = self.i_att(img_feat, ques_feat)     # [batch, 1024]

        if self.__C.HIGH_ORDER:  # MFH 这里if不成立
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z=[batch, 1, 1000], _=[batch, 1, 5000]
            z = z.squeeze(1)                                                            # [batch, 1000]

        return z          # [batch, 1000]
