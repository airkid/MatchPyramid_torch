import codecs
from torch.utils.data import Dataset


class MSRPDataset(Dataset):

    def __init__(self, data_dir, data_type="train"):
        self.data_list = list()
        _file = codecs.open(data_dir+"_"+data_type+".txt", 'r', 'utf-8')
        for line in _file.readlines():
            _, _, _, sen1, sen2, label = line.split("\t")
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


def truncate(sen1, len1, sen2, len2, label, max_seq_len=128):
    ###
    return sen1, len1, sen2, len2, label