# simple_bert_for_tf2
Build bert as a keras layer using TF2.0 .

## Itroduction
Bert model is build as a Keras Layer.

merit:
-


## Files
|--bert_parts
|    |--layers.py       bert layer using keras TF2 | 基于keraslayer的bert layer
|    |--tokenizer.py    tokenizer for chinese      | 用于对中文做tokenize的文件
|    |--vocab.txt       vocab file                 | 词典文件，用于将字符转换为token id
|--datasource.py        genarate data              | 产生数据
|--finetune.py          an example for finetune    | 微调的例子
|--pretrain.py          an example for pretrain    | 预训练的例子

