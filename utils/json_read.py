import json
import os
import numpy as np
import pandas as pd
# with open("G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\data2020/valset.json") as f1:
#     label = json.load(f1)
#     ans_list = []
#     for i in range(len(label)):
#         ans = label[i]['answer']
#         ans_list.append(ans)
#
# with open(r'G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BBNBioBertInference\results\val\resnest-crop-retrieval/prediction.txt') as f2:
#     pred = f2.readlines()
#     pred_list = []
#     for i in range(len(pred)):
#         pred_res = pred[i].split('|')[1].strip()
#         pred_list.append(pred_res)
#
# acc = 0
# for i in range(len(ans_list)):
#     if ans_list[i] == pred_list[i]:
#         acc += 1
# print(acc/len(ans_list))

# with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\data2020/trainset.json') as f:
#     p1 = json.load(f)
#
# with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\data2020/full_VQA_Med_2020_train.json') as g:
#     p2 = json.load(g)
#
# with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BioBert\data/train0set.json') as a:
#     q1 = json.load(a)
#
# with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BioBert\data/train1set.json') as b:
#     q2 = json.load(b)
#
# with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BioBert\data/train2set.json') as c:
#     q3 = json.load(c)
#
#
# for i in range(len(p2)):
#     sample = p2[i]
#     print("debug")

# train_close_answer_dict = {}
# close_ques_list = []
# close_ans_list = []
# with open('G:\PythonWorkplace\VQA-master\data/trainset_rad.json') as t:
#     trainset = json.load(t)
#     for train_entity in trainset:
#         question = train_entity['question']
#         answer = train_entity['answer']
#         answer_type = train_entity['answer_type']
#         if answer_type == 'CLOSED' and answer not in close_ans_list:
#             close_ans_list.append(answer)
#         if answer_type == 'CLOSED' and question not in close_ques_list:
#             close_ques_list.append(question)
#         if answer_type == 'CLOSED':
#             train_close_answer_dict[question] = answer
# with open("G:\PythonWorkplace\ImageCLEF2021\data2020/Rad.json", "w") as f:
#     f.write(json.dumps(train_close_answer_dict, ensure_ascii=False, indent=4, separators=(',', ':')))


import torch
import clip
print(clip.available_models())
