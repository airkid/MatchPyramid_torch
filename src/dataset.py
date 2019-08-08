import codecs
from torch.utils.data import Dataset


class MSRPDataset(Dataset):

    def __init__(self, data_dir, data_type="train"):
        self.data_list = list()
        _file = codecs.open(data_dir+"_"+data_type+".txt", 'r', 'utf-8')
        for line in _file.readlines():
            query, title, label = line.split("\t")
        _file.close()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
