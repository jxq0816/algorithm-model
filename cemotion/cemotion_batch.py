#!/usr/bin/python
# -*- coding:utf-8 -*-
from cemotion import Cemotion
import pandas as pd


def text_emotion(input_filename,output_filename):
    c = Cemotion()
    cmt_data = pd.read_csv(input_filename, sep='\t', header=None, names=['cmt_id', 'url', 'cmt_content_base', 'cmt_content'])
    file = open(output_filename, 'w', encoding='utf-8')
    for i,colomn in cmt_data.iterrows():
        cmt_id = colomn['cmt_id']
        url = colomn['url']
        cmt_content_base = colomn['cmt_content_base']
        cmt_content = colomn['cmt_content']
        rs = str(cmt_id) + '\t' + str(url) + '\t' + str(cmt_content_base) + '\t' + str(cmt_content)
        val = c.predict(cmt_content)
        if val > 0.5:
            rs = rs + '\t'+'正向'+'\n'
        else:
            rs = rs + '\t'+'负向'+'\n'
        file.write(rs)
    file.close()


if __name__ == "__main__":

    text_emotion('cmt2.txt','cmt_rs2.txt')