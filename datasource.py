import tensorflow as tf
import numpy as np
from bert_parts import tokenizer


class FakeDataGen(tf.keras.utils.Sequence):
    def __init__(self,
                 seq_max_len,
                 data_len=10,
                 batch_size=5):
        self.seq_max_len = seq_max_len
        self.data_len = data_len
        self.batch_size = batch_size
        self.data = ['蒙大拿蒙大拿蒙大拿蒙大拿', '大选帝侯', '中途岛', '莫斯科', '基林',
                     '岛风', '哈巴罗夫斯克', '得梅因', '兴登堡', '藏']

    def __len__(self):
        return int(np.ceil(self.data_len/self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = [tokenizer.str2idlist(x, self.seq_max_len) for x in batch_data]
        batch_x, batch_mask = tokenizer.AddMask(batch_data, self.seq_max_len)
        return [batch_x, batch_x, batch_mask]


def fake_data_gen_2(seq_max_len=100):
    data = ['蒙大拿蒙大拿蒙大拿蒙大拿', '大选帝侯', '中途岛', '莫斯科莫斯科', '基林',
            '岛风shimakaze', '哈巴罗夫斯克', '得梅因', '兴登堡', '藏']
    data = [tokenizer.str2idlist(x, seq_max_len) for x in data]
    label = [0, 1, 2, 0, 1, 0, 2, 1, 1, 0]
    return np.array(data), np.array(label)


if __name__ == '__main__':
    fake_data_gen_2()
