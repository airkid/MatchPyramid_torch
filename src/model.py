import json
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.dataset import collate_fn, truncate
from src.utils import to_cuda

logger = getLogger()


class MatchPyramidClassifier(object):

    def __init__(self, params):
        logger.info("Initializing MatchPyramidClassifier")
        self.params = params
        self.train_data = params.train_data
        self.test_data = params.test_data
        self.epoch_cnt = 0

        self.embedding = torch.nn.Embedding.from_pretrained(params.glove_weight)

        self.matchPyramid = MatchPyramid(self.params)

        self.optimizer = torch.optim.Adam(
            list(self.embedding.parameters())+list(self.matchPyramid.parameters()),
            lr=self.params.lr
        )
        self.embedding.cuda()
        self.matchPyramid.cuda()

    def run(self):
        for i in range(self.params.n_epochs):
            self.train()
            self.evaluate()
            self.epoch_cnt += 1

    def train(self):
        logger.info("Training in epoch %i" % self.epoch_cnt)
        self.embedding.train()
        self.matchPyramid.train()
        data_loader = DataLoader(self.train_data,
                                 batch_size=self.params.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn)
        for data_iter in data_loader:
            sen1, len1, sen2, len2, label = data_iter
            sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = truncate(
                sen1, len1, sen2, len2, label,
                self.params.word2idx,
                max_seq_len=self.params.max_seq_len)
            sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = to_cuda(
                sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts)
            sen1_embedding = self.embedding(sen1_ts)
            sen2_embedding = self.embedding(sen2_ts)
            mp_output = self.matchPyramid(sen1_embedding, sen2_embedding)
            loss = F.cross_entropy(mp_output, label_ts)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        logger.info("Evaluating in epoch %i" % self.epoch_cnt)
        self.embedding.eval()
        self.matchPyramid.eval()
        data_loader = DataLoader(self.test_data,
                                 batch_size=self.params.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
        pred_list = list()
        label_list = list()
        with torch.no_grad():
            for data_iter in data_loader:
                sen1, len1, sen2, len2, label = data_iter
                sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = truncate(
                    sen1, len1, sen2, len2, label,
                    self.params.word2idx,
                    max_seq_len=self.params.max_seq_len)
                sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = to_cuda(
                    sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts)
                sen1_embedding = self.embedding(sen1_ts)
                sen2_embedding = self.embedding(sen2_ts)
                mp_output = self.matchPyramid(sen1_embedding, sen2_embedding)
                predictions = mp_output.data.max(1)[1]
                pred_list.extend(predictions.tolist())
                label_list.extend(label_ts.tolist())
        acc = accuracy_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)
        logger.info("ACC score in epoch %i :%.4f" % (self.epoch_cnt, acc))
        logger.info("F1 score in epoch %i :%.4f" % (self.epoch_cnt, f1))


class MatchPyramid(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.max_len1 = params.max_seq_len
        self.max_len2 = params.max_seq_len
        self.conv1_size = [int(_) for _ in params.conv1_size.split("_")]
        self.pool1_size = [int(_) for _ in params.pool1_size.split("_")]
        self.conv2_size = [int(_) for _ in params.conv2_size.split("_")]
        self.pool2_size = [int(_) for _ in params.pool2_size.split("_")]
        self.dim_hidden = params.mp_hidden

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.conv1_size[-1],
                                     kernel_size=tuple(
                                         self.conv1_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_size[-1],
                                     out_channels=self.conv2_size[-1],
                                     kernel_size=tuple(
                                         self.conv2_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        self.linear1 = torch.nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv2_size[-1],
                                       self.dim_hidden, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = torch.nn.Linear(self.dim_hidden, params.dim_out, bias=True)
        # torch.nn.init.kaiming_normal_(self.linear2.weight)
        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": [self.conv1_size, self.conv2_size],
                 "Pooling": [self.pool1_size, self.pool2_size],
                 "MLP": self.dim_hidden}))

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
            simi_img = F.pad(simi_img, (0, pad2, 0, pad1))
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


