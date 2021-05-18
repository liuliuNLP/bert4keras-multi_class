# -*- coding: utf-8 -*-
import numpy as np
import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model  # 必须保留
from bert4keras.snippets import sequence_padding

maxlen = 300
set_gelu('tanh')  # 切换gelu版本

config_path = 'electra_180g_small/small_discriminator_config.json'
checkpoint_path = 'electra_180g_small/electra_180g_small.ckpt'
dict_path = 'electra_180g_small/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)
model_path = 'electra_180g_small-model.h5'
model = keras.models.load_model(model_path)


def pre_model(text):
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    token_ids, segment_ids = tokenizer.encode(first_text=text, maxlen=maxlen)
    token_list = sequence_padding([token_ids])
    segment_list = sequence_padding([segment_ids])
    label = model.predict([np.array(token_list), np.array(segment_list)]).argmax(axis=1)
    return int(label[0])


if __name__ == '__main__':
    label_dict = {0: "体育", 1: "健康", 2: "教育", 3: "汽车", 4: "军事"}
    text = '继成功入选国家队后，解立彬又荣获05-06赛季CBA联赛最佳新人，可谓双喜临门。解立彬曾效力大超联赛中国人民大学队，在首届联赛中表现出色，帮助人大获得大超总冠军，随后登陆CBA赛场，为北京首钢队攻城拔寨，最终随队跻身四强。短短6个月间，他从一名大学生球员转变为职业球员，并入选国家队18人大名单，是大超联赛走出的首位成年国手。在他当选最佳新人之际，祝愿他在国家队有好的表现，为国争光，为大超联赛争光。'
    label = pre_model(text)
    print(label_dict.get(label))
