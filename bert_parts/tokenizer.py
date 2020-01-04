# -*- coding: utf-8 -*-
import numpy as np

with open('bert_parts/vocab.txt', 'r', encoding='utf-8') as handle:
    vocab = handle.readlines()
vocab = [x.strip() for x in vocab]
vocab2id_dic = dict(zip(vocab, list(range(len(vocab)))))
id2vocab_dic = dict(zip(list(range(len(vocab))), vocab))


def truncate(text, seq_max_len):
    """
    截断，因为开头留给cls，所以输入的最大长度是maxlen-1
    :param text: 输入的string
    :param seq_max_len: token ids的长度，也是模型允许输入的最大长度
    :return: 截断后的string
    """
    if len(text) > seq_max_len - 1:
        return text[:seq_max_len-1]
    else:
        return text


def decorate(char_list):
    """
    加上cls，根据场景不同加的标识也不同，比如两个句子的就要加sep。根据场景不同可以更改
    :param char_list: 由char组成的list
    :return: 加上[cls]等标识后的char list
    """
    char_list = ['[CLS]'] + char_list
    return char_list


def padding(char_list, seq_max_len):
    """
    没到seq max len的句子，用pad补全
    :param char_list: 由char组成的list
    :param seq_max_len: token ids的长度，也是模型允许输入的最大长度
    :return: padding后的char list
    """
    char_list = char_list + ['[PAD]'] * (seq_max_len - len(char_list))
    return char_list


def vocab2id(char_list):
    """
    字符的list，按照词表字典，转换成id的list
    :param char_list: 由char组成的list
    :return:由id组成的list
    """
    id_list = []
    for i in range(len(char_list)):
        if char_list[i] in vocab2id_dic.keys():
            id_list.append(vocab2id_dic.get(char_list[i]))
        else:
            id_list.append(vocab2id_dic.get('[UNK]'))
    return id_list


def id2vocab(id_list):
    """
    id的list，按照词表字典的反向，转换成字符的list
    :param id_list:
    :return:
    """
    char_list = []
    for i in range(len(id_list)):
        if id_list[i] in id2vocab_dic.keys():
            char_list.append(id2vocab_dic.get(id_list[i]))
        else:
            char_list.append('*')
    return char_list


def str2idlist(text, seq_max_len):
    """
    字符串转换成id的list
    :param text: string
    :param seq_max_len: token ids的长度，也是模型允许输入的最大长度
    :return: id组成的一个list
    """
    text = truncate(text, seq_max_len)
    char_list = list(text)
    char_list = decorate(char_list)
    char_list = padding(char_list, seq_max_len)
    id_list = vocab2id(char_list)
    return id_list


def idlist2str(id_list):
    """
    id的list，转换成字符后，拼接成字符串
    :param id_list: id组成的list
    :return: string
    """
    char_list = id2vocab(id_list)
    print(char_list)
    str = ''
    for x in char_list:
        if x not in {'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'}:
            str += x
    return str


def AddMask(inputs, seq_max_len, mask_rate=0.15, mask_value=4):
    """
    对token ids做随机15%的mask。[MASK]的id默认为4。
    :param inputs: numpy array的输入，[batch_size, seq_man_len]
    :param seq_max_len: token ids的长度，也是模型允许输入的最大长度
    :param mask_rate: 随机mask的比例，默认为15%
    :param mask_value: [MASK]在vocab中的id，默认是4
    :return:mask后的inputs[batch_size, seq_man_len]，mask的位置[batch_size, seq_man_len]
    """
    padding_mask = (np.array(inputs) > 0).astype(int)  # 不是pad的部分都标为1，pad的都是0
    padding_sum = np.sum(padding_mask, axis=1)   # 把不是pad的部分count一下（包含了cls）
    mask_char_num = np.ceil(padding_sum * mask_rate).astype(int)
    mask_list = []
    for i in range(len(mask_char_num)):
        if padding_sum[i] == 2:   # 除了cls以外，只有一个字符，就不pad了
            posi = np.random.choice(a=np.arange(1, padding_sum[i]), size=0, replace=False)
            mask_list.append(np.sum(np.eye(seq_max_len)[posi], axis=0))
        else:
            posi = np.random.choice(a=np.arange(1, padding_sum[i]), size=mask_char_num[i], replace=False)
            mask_list.append(np.sum(np.eye(seq_max_len)[posi], axis=0))
    mask_list = np.array(mask_list)
    mask_value_matrix = mask_list * mask_value   # 把该mask的地方都变成mask的token id
    masked_inputs = (mask_list == 0).astype(int)
    outputs = (inputs * masked_inputs + mask_value_matrix).astype(int)
    return outputs, mask_list


if __name__ == '__main__':
    text = '0001威猛先生厨房重油污净双包装500g洗洁精225g12'
    print('input:')
    print(text)
    print()

    id_l = str2idlist(text, seq_max_len=10)
    print('string to ids list, seq_max_len = 10:')
    print(id_l)
    print()

    id_l = str2idlist(text, seq_max_len=100)
    print('string to ids list, seq_max_len = 100:')
    print(id_l)
    print()

    masked_id_l, mask = AddMask([id_l], seq_max_len=100)
    print('masked id list, seq_max_len = 100:')
    print(masked_id_l)
    print('mask:')
    print(mask)
    print()

    masked_id_l_to_char = id2vocab(masked_id_l[0])
    print('masked id list to char list:')
    print(masked_id_l_to_char)
