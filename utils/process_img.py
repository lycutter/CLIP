import cv2
import json
import pickle
import numpy as np

img_dir = r'G:\PythonWorkplace\HCP-MIC-at-ImageCLEF-VQA-Med-2020-master\data2020\val2020\images/'

imgid2idx = json.load(open('../data2020/imgid2idx_val.json'))

idx2imgid =  {value:key for key, value in imgid2idx.items()}
img_list = []
for i in range(len(imgid2idx)):
    img_path = img_dir + idx2imgid[i]
    # img = cv2.imread(img_dir + idx2imgid[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_dir + idx2imgid[i])
    width = 224
    height = 224
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img_list.append(img)
img_list = np.array(img_list)
pickle.dump(img_list, open('../data2020/val_img_224x224.pkl', 'wb')) #后缀.pkl可加可不加