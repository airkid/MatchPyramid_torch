import json
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F


logger = getLogger()


class MatchPyramidClassifier(object):

    def __init__(self, params):
        self.params = params
        self.train_data = params.train_data
        self.test_data = params.test_data

        self.embedding = torch.nn.Embedding()
        self.matchPyramid = MatchPyramid(self.params)

    def run(self):
        for i in range(self.params.n_epochs):
            self.train()
            self.evaluate()

    def train(self):
        self.embedding.train()
        self.matchPyramid.train()

    def evaluate(self):
        self.embedding.eval()
        self.matchPyramid.eval()
        with torch.no_grad():
            pass


class MatchPyramid(torch.nn.Module):

    def __init__(self, max_query, max_title, num_class, logger=None):
        super().__init__()
        self.max_len1 = max_query
        self.max_len2 = max_title

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=MPConfig.kernel[0][-1],
                                     kernel_size=tuple(
                                         MPConfig.kernel[0][:-1]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=MPConfig.kernel[0][-1],
                                     out_channels=MPConfig.kernel[1][-1],
                                     kernel_size=tuple(
                                         MPConfig.kernel[1][:-1]),
                                     padding=0,
                                     bias=True
                                     )
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(MPConfig.pool[0]))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(MPConfig.pool[1]))
        self.linear1 = torch.nn.Linear(MPConfig.pool[1][0] * MPConfig.pool[1][1] * MPConfig.kernel[1][-1],
                                       MPConfig.mlp_hidden, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = torch.nn.Linear(MPConfig.mlp_hidden, num_class, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear2.weight)
        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": MPConfig.kernel, "Pooling": MPConfig.pool, "MLP": MPConfig.mlp_hidden}))

    def forward(self, x1, x2):
        # x1,x2:[batch, seq_len, dim_xlm]
        bs, seq_len1, dim_xlm = x1.size()
        seq_len2 = x2.size()[1]
        pad1 = self.max_len1 - seq_len1
        pad2 = self.max_len2 - seq_len2
        # simi_img:[batch, 1, seq_len, seq_len]
        # x1_norm = x1.norm(dim=-1, keepdim=True)
        # x1_norm = x1_norm + 1e-8
        # x2_norm = x2.norm(dim=-1, keepdim=True)
        # x2_norm = x2_norm + 1e-8
        # x1 = x1 / x1_norm
        # x2 = x2 / x2_norm
        # use cosine similarity since dim is too big for dot-product
        simi_img = torch.matmul(x1, x2.transpose(1, 2)) / np.sqrt(dim_xlm)
        if pad1 != 0 or pad2 != 0:
            simi_img = F.pad(simi_img, [0, pad2, 0, pad1])
        assert simi_img.size() == (bs, self.max_len1, self.max_len2)
        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        # [batch, 1, conv1_w, conv1_h]
        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        simi_img = self.pool2(simi_img)
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        simi_img = simi_img.squeeze(1).view(bs, -1)
        # output = self.linear1(simi_img)
        output = self.linear2(F.relu(self.linear1(simi_img)))
        return output


class MPConfig(object):
    kernel = [[5, 5, 64], [3, 3, 32]]
    pool = [(14, 32), (4, 10)]
    mlp_hidden = 512


if __name__ == "__main__":
    mp = MatchPyramid(16, 32, 4)
    input1 = torch.randn(24, 16, 100)
    input2 = torch.randn(24, 32, 100)
    output = mp(input1, input2)
    print(output.size())

