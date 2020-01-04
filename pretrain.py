import tensorflow as tf
from bert_parts import layers
import datasource

config = {
    'seq_max_len': 100,
    'vocab_size': 7364,
    'embedding_size': 128,
    'num_transformer_layers': 6,
    'num_attetion_heads': 8,
    'intermediate_size': 32
}


class MLMLoss(tf.keras.layers.Layer):
    """设置mlm预训练任务的loss。
    mlm任务的输出为模型输出接一个dense然后再过模型中token tmbedding的转置，变回为word的onehot格式
    loss就是这个onehot和原本的onehot格式计算多分类交叉熵"""
    def __init__(self, **kwargs):
        super(MLMLoss, self).__init__(**kwargs)

    def call(self, inputs):
        y_true, y_pred, mlm_mask = inputs
        mlm_mask_re = tf.keras.backend.cast(tf.keras.backend.greater(y_true, 4), 'float32')
        loss = tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred)
        loss1 = tf.keras.backend.sum(loss * mlm_mask, axis=1) / (tf.keras.backend.sum(mlm_mask, axis=1) + 1)
        loss2 = tf.keras.backend.sum(loss * mlm_mask_re, axis=1) / (tf.keras.backend.sum(mlm_mask_re, axis=1) + 1)
        loss2 = loss2 / 2
        self.add_loss(loss1+loss2, inputs=True)
        self.add_metric(loss2+loss2, aggregation="mean", name="mlm_loss")
        return loss1+loss2


def build_model():
    # inputs
    input_data = tf.keras.Input(shape=(config['seq_max_len'],))
    label = tf.keras.Input(shape=(config['seq_max_len'],))
    mlm_mask = tf.keras.Input(shape=(config['seq_max_len'],))
    # model
    bert_layer = layers.BertLayer(vocab_size=config['vocab_size'],
                                  embedding_size=config['embedding_size'],
                                  num_transformer_layers=config['num_transformer_layers'],
                                  num_attention_heads=config['num_attetion_heads'],
                                  intermediate_size=config['intermediate_size'])
    bert_out = bert_layer(input_data)
    # outputs
    bert_out_to_token = layers.TransTokenEmbedding(bert_layer.weights[0])(bert_out)
    # loss
    loss = MLMLoss()([label, bert_out_to_token, mlm_mask])
    # build model
    pretrained_model = tf.keras.models.Model(inputs=[input_data, label, mlm_mask], outputs=[loss])
    pretrained_model.summary()
    pretrained_model.compile(optimizer='adam')
    return pretrained_model


def bert_layer_weights(filename='pretrain_weights.h5'):
    """
    读取保存的预训练模型中的bert layer的weights
    注意是按照当前的预训练模型来读取的，模型结构要一致。
    model.layers[1]是bert layer
    :param filename: 预训练模型权重的储存文件名（当前目录下）
    :return: bert layer的权重（按照预训练模型中的config）
    """
    pretrained_model = build_model()
    pretrained_model.load_weights(filename)
    # print(pretrained_model.layers[1])   # bert_parts.layers.BertLayer
    return pretrained_model.layers[1].get_weights()


if __name__ == '__main__':
    fake_data = datasource.FakeDataGen(seq_max_len=config['seq_max_len'])
    pretrained_model = build_model()
    pretrained_model.fit_generator(fake_data, epochs=50)
    pretrained_model.save_weights('pretrain_weights.h5', overwrite=True)

    # 测试获取bert layert的weights
    print(bert_layer_weights())
