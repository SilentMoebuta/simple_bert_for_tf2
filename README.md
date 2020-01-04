# simple_bert_for_tf2
Build bert as a keras layer using TF2.0 .

## Itroduction
Bert model is build as a Keras Layer.
merits
1.Easy to apply bert as a layer in a practical TF2.0 model.
2.Using only numpy and Tensorflow2.0 as third party packages.
notes

## Files
```
|--bert_parts
|    |--layers.py       bert layer using keras TF2 | 基于keraslayer的bert layer
|    |--tokenizer.py    tokenizer for chinese      | 用于对中文做tokenize的文件
|    |--vocab.txt       vocab file                 | 词典文件，用于将字符转换为token id
|--datasource.py        genarate data              | 产生数据
|--finetune.py          an example for finetune    | 微调的例子
|--pretrain.py          an example for pretrain    | 预训练的例子
```
