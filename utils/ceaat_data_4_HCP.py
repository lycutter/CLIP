import PIL.Image as Image
import os
import json
import _pickle as cPickle

# for classification: 330 diseases
# for VQA: 332 candidate answers (normal + unclear_abnormal + 330 dieases)
# Do remember to modify the cache path, image dir, json path, output_name and closed_answer_num

# json_path = '/home/duadua/MVQA/SMMC/data/valset.json'  # txt2json.py
# cache_path = '/home/duadua/MVQA/HPC-MIC at ImageCLEF VQA-Med 2020/BBN/cache4VQA-Med'  # create_cache4VQA.py
# img_dir = '/home/duadua/MVQA/SMMC/data/val2020/images/'
# output_name = './jsons/VQA_Med_1920_val.json'

json_path = 'G:/PythonWorkplace/HCP-MIC-at-ImageCLEF-VQA-Med-2020-master/data2020/valset.json'
cache_path = 'G:/PythonWorkplace/HCP-MIC-at-ImageCLEF-VQA-Med-2020-master/data2020/cache4VQA-Med/trainval_ans2label.pkl'
img_dir = 'G:/PythonWorkplace/HCP-MIC-at-ImageCLEF-VQA-Med-2020-master/data2020/val2020/images/'
output_name = 'G:/PythonWorkplace/HCP-MIC-at-ImageCLEF-VQA-Med-2020-master/data2020/full_VQA_Med_2020_val.json'


# closed_answer_num = 28


ans2label = cPickle.load(open(cache_path, 'rb'))
cnt = 0
valid = 0
data_dict = {'annotations': [], 'num_classes': 332}
file = json.load(open(json_path))

flag = 'val'


with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BioBert\data/train0set.json') as a:
    q1 = json.load(a)

with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BioBert\data/train1set.json') as b:
    q2 = json.load(b)

with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BioBert\data/train2set.json') as c:
    q3 = json.load(c)



with open('G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\BioBert\data/valset.json') as d:
    q_valset = json.load(d)


q_label_dict = {
    '0': [1, 0, 0],
    '1': [0, 1, 0],
    '2': [0, 0, 1]
}
for item in file:

    ques = item['question']
    q_label = None

    if flag == 'train':
        for i in range(len(q1)):
            q_tmp = q1[i]['question']
            if ques == q_tmp:
                q_label = 0
                break
        for i in range(len(q2)):
            q_tmp = q2[i]['question']
            if ques == q_tmp:
                q_label = 1
                break
        for i in range(len(q3)):
            q_tmp = q3[i]['question']
            if ques == q_tmp:
                q_label = 2
                break
    elif flag == 'val':
        for i in range(len(q_valset)):
            q_tmp = q_valset[i]['question']
            if ques == q_tmp:
                q_label = q_valset[i]['label']
                break

    img_name, ans = item['image_name'][:-4], item['answer']
    sample_dict = {}

    sample_dict['image_id'] = valid

    sample_dict['fpath'] = img_dir + img_name + '.jpg'
    # sample_dict['fpath'] = '/home/liyong/PythonWorkPlace/HCP-MIC-at-ImageCLEF-VQA-Med-2020-master/data2020/val2020/images/' + img_name + '.jpg'

    sample_dict['question'] = ques

    sample_dict['q_label'] = q_label

    sample_dict['answer'] = item['answer']

    sample_dict['qid'] = item['qid']

    img_path = os.path.join(img_dir, img_name + '.jpg')
    img = Image.open(img_path)
    w, h = img.size

    sample_dict['im_height'] = h
    sample_dict['im_width'] = w
    sample_dict['category_id'] = ans2label[ans]
    data_dict['annotations'].append(sample_dict)

    valid += 1

with open(output_name, 'w') as f:
    json.dump(data_dict, f)
