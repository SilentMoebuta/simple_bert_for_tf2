import tensorflow as tf
import pretrain
import datasource
from bert_parts import layers

# config和pretrain的是一样时，可以读取预训练模型的bert层参数
config = {
    'seq_max_len': 100,
    'vocab_size': 7364,
    'embedding_size': 128,
    'num_transformer_layers': 6,
    'num_attetion_heads': 8,
    'intermediate_size': 32
}

# 数据准备
train_x, train_y = datasource.fake_data_gen_2(seq_max_len=config['seq_max_len'])

# bert层构建，和预训练模型中的相同
tiny_bert_layer = layers.BertLayer(vocab_size=config['vocab_size'],
                                   embedding_size=config['embedding_size'],
                                   num_transformer_layers=config['num_transformer_layers'],
                                   num_attention_heads=config['num_attetion_heads'],
                                   intermediate_size=config['intermediate_size'])

# 读取预训练模型中的bert层权重
bert_layer_weights = pretrain.bert_layer_weights(filename='pretrain_weights.h5')

# 模型
model = tf.keras.models.Sequential([
    tiny_bert_layer,
    tf.keras.layers.Lambda(lambda seq: seq[:, 0, :]),
    tf.keras.layers.Dense(units=3, activation='softmax')
])
model.build(input_shape=(None, 100))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy())


model.fit(train_x, train_y,
          batch_size=5,
          epochs=20,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                                                      restore_best_weights=True)])
model.save_weights('finetune_weights.h5', overwrite=True)
