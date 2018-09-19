#!/usr/bin/python
 # -*- coding: utf-8 -*-
from pathlib import Path


def read_weishi(path_in_list,out_file,label_file):
    for index in range(0,len(path_in_list)):
        p = Path(path_in_list[index])
        file_names = list(p.glob('*.txt'))
        text_list=[]
        for file_name in file_names:
            with open(str(file_name),mode='rb') as f:
                content = f.read()
                str_content =content.decode('utf-8')
                print(str_content)
                text_list.append(str_content.strip())
        print(text_list)
        with open(out_file, 'a', encoding='utf-8') as fout:
            with open(label_file, 'a') as label_out:
                for ite in text_list:
                    fout.write(ite)
                    fout.write('\n')
                    label_out.write(str(index))
                    label_out.write('\n')






path_in = ['E:\\ads_data\\normal\\normal_text\\','E:\\ads_data\\ads\\ads_text\\','E:\\ads_data\\anc\\anc_text\\']
f_out='../data/ads_data.txt'
label_data='../data/ads_label.txt'

read_weishi(path_in,f_out,label_data)


