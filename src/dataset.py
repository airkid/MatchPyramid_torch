import codecs
import torch
from torch.utils.data import Dataset


class MSRPDataset(Dataset):

    def __init__(self, data_dir, data_type="train"):
        self.data_list = list()
        _file = codecs.open(data_dir+"_"+data_type+".txt", 'r', 'utf-8')
        for line in _file.readlines()[1:]:
            label, _, _, sen1, sen2 = line.split("\t")
            sen1 = sen1.strip().split(" ")
            sen2 = sen2.strip().split(" ")
            label = int(label.strip())
            self.data_list.append((sen1, sen2, label))
        _file.close()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(data):
    sen1_list, sen2_list, label_list = list(), list(), list()
    for data_item in data:
        sen1_list.append(data_item[0])
        sen2_list.append(data_item[1])
        label_list.append(data_item[2])
    len1 = [len(sen) for sen in sen1_list]
    len2 = [len(sen) for sen in sen2_list]
    return sen1_list, len1, sen2_list, len2, label_list


def truncate(sen1, len1, sen2, len2, label, word2idx, max_seq_len=32):
    def get_idx(w):
        if w in word2idx:
            return word2idx[w]
        else:
            return word2idx["UNK"]
    batch_size = len(sen1)
    max_len1 = min(max(len1), max_seq_len)
    max_len2 = min(max(len2), max_seq_len)
    len1 = torch.LongTensor(len1)
    len2 = torch.LongTensor(len2)
    label = torch.LongTensor(label)
    sen1_ts = torch.LongTensor(batch_size, max_len1).fill_(0)
    sen2_ts = torch.LongTensor(batch_size, max_len2).fill_(0)
    for i in range(batch_size):
        if len1[i] > max_seq_len:
            len1[i] = max_seq_len
        _sent1 = torch.LongTensor([get_idx(w) for w in sen1[i]])
        sen1_ts[i, :len1[i]] = _sent1[:len1[i]]
        if len2[i] > max_seq_len:
            len2[i] = max_seq_len
        _sent2 = torch.LongTensor([get_idx(w) for w in sen2[i]])
        sen2_ts[i, :len2[i]] = _sent2[:len2[i]]
    return sen1_ts, len1, sen2_ts, len2, label
