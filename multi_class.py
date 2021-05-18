# -*- coding: utf-8 -*-
import numpy as np
import json
import pandas as pd
import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
from bert4keras.tokenizers import Tokenizer
from sklearn.model_selection import train_test_split
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import tensorflow as tf
from bert4keras.backend import set_gelu
from sklearn.metrics import classification_report

set_gelu('tanh')  # 切换gelu版本

num_classes = 5
maxlen = 300
batch_size = 32
number_of_epochs = 50

# ###########  albert  ##############
# config_path = 'albert_small_zh_google/albert_config_small_google.json'
# checkpoint_path = 'albert_small_zh_google/albert_model.ckpt'
# dict_path = 'albert_small_zh_google/vocab.txt'
# model_type = 'albert'
# model_name = 'albert_small_zh_google'

###########  electra_180g_small  ##############
config_path = 'electra_180g_small/small_discriminator_config.json'
checkpoint_path = 'electra_180g_small/electra_180g_small.ckpt'
dict_path = 'electra_180g_small/vocab.txt'
model_type = 'electra'
model_name = 'electra_180g_small'

# ###########   bert  ##############
# config_path = './chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
# model_type = 'bert'
# model_name = 'bert'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def split_dataset(df):
    train_set, x = train_test_split(df, stratify=df['new_label'], test_size=0.3, random_state=42)
    eval_set, test_set = train_test_split(x, stratify=x['new_label'], test_size=0.5, random_state=43)
    return train_set, eval_set, test_set


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def get_train_data(df_raw):
    train_data, eval_data, test_data = split_dataset(df_raw)
    train_data = [(row["text"], row["new_label"]) for index, row in train_data.iterrows()]
    eval_data = [(row["text"], row["new_label"]) for index, row in eval_data.iterrows()]
    test_data = [(row["text"], row["new_label"]) for index, row in test_data.iterrows()]

    train_encoded = data_generator(train_data, batch_size)
    eval_encoded = data_generator(eval_data, batch_size)
    test_encoded = data_generator(test_data, batch_size)
    return train_encoded, eval_encoded, test_encoded


def standard_label(df=None):
    key_list = [i for i in df["new_label"].tolist()]
    value_list = df["label"].tolist()
    with open("label.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(zip(key_list, value_list)), ensure_ascii=False, indent=2))


def train_model(df_data):
    train_dataset, eval_dataset, test_dataset = get_train_data(df_data)
    gpu_len = len(tf.config.experimental.list_physical_devices('GPU'))
    print("gpu_len:" + str(gpu_len))
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model=model_type,
            return_keras_model=False,
        )
        output = Lambda(lambda x: x[:, 0])(bert.model.output)
        output = Dense(
            units=num_classes,
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)
        model = tf.keras.models.Model(bert.model.input, output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        model.summary()

    model.fit(
        train_dataset.forfit(),
        steps_per_epoch=len(train_dataset),
        epochs=number_of_epochs,
        validation_data=eval_dataset.forfit(),
        validation_steps=number_of_epochs
    )
    print("evaluate test_set:", model.evaluate(test_dataset.forfit(), steps=number_of_epochs))
    model.save('{}-model.h5'.format(model_name))
    evaluate_report(df_data)


def evaluate_report(df_data):
    model = tf.keras.models.load_model('{}-model.h5'.format(model_name))
    true_y_list = [i for i in df_data["new_label"].tolist()]
    pred_y_list = []
    for text in df_data["text"].tolist():
        tokenizer = Tokenizer(dict_path, do_lower_case=True)
        token_ids, segment_ids = tokenizer.encode(first_text=text, maxlen=maxlen)
        token_list = sequence_padding([token_ids])
        segment_list = sequence_padding([segment_ids])
        label = model.predict([np.array(token_list), np.array(segment_list)]).argmax(axis=1)
        pred_y_list.append(label[0])

    with open("label.json", "r", encoding="utf-8") as f:
        labels = json.loads(f.read())
    target_name_list = list(labels.values())
    report = classification_report(true_y_list, pred_y_list, target_names=target_name_list, digits=4, output_dict=True)
    print(report)
    df = pd.DataFrame(report).transpose()
    df.to_csv("{}-report.csv".format(model_type), encoding='utf_8_sig', index=True)


if __name__ == '__main__':
    data = pd.read_excel(r'data/data.xlsx')
    label_list = list(set(data["label"].tolist()))
    df_label = pd.DataFrame({"label": label_list, "new_label": list(range(len(label_list)))})
    standard_label(df_label)
    df_data = pd.merge(data, df_label, on="label", how="left")
    df_data = df_data.sample(frac=1)
    train_model(df_data)
