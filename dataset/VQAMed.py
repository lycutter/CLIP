from dataset.baseset import BaseSet
import random
from dataset.nlp_dataset import Dictionary
import torch
import numpy as np


class VQAFeatureDataset(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super().__init__(mode, cfg, transform)

        # load img
        random.seed(0)

        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode=="train"):
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            self.class_dict = self._get_class_dict()

        # load nlp
        dictionary = Dictionary.load_from_file(cfg.dictionary_path)
        self.dictionary = dictionary
        self.tokenize(12)
        self.tensorize()

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.data:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            assert len(tokens) == max_length
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.data:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            label = torch.from_numpy(np.array(entry['q_label'])).long()
            # label = entry['q_label']
            # label = torch.tensor(label, dtype=torch.int64).cuda()
            entry['q_label'] = label

    def __getitem__(self, index):


        if self.cfg.train_sample_type == "weighted sampler" and self.mode == 'train': # if不成立
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)


        now_info = self.data[index]


        # get img
        img = self._get_image(now_info)
        image = self.transform(img)

        # if self.mode == 'valid':
        #     print("image.shape: {}".format(image.shape))

        meta = dict()
        if self.dual_sample:
            if self.cfg.train_sample_dual_sample_type == "reverse":
                sample_class = self.sample_class_index_by_weight() # 随机抽出一个类别的answers
            elif self.cfg.train_sample_dual_sample_type == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            if sample_class not in self.class_dict:
                sample_class = 1
            sample_indexes = self.class_dict[sample_class] # class_dict是一个category为key, 样本的index_list为value的dict, 因此这返回为第sample_class个answer的全部样本的list
            sample_index = random.choice(sample_indexes) # 随机取一个该answer的样本出来
            sample_info = self.data[sample_index] # 获取一个样本
            sample_img, sample_label = self._get_image(sample_info), sample_info['category_id'] # 获取图片和label
            sample_img = self.transform(sample_img)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.mode != 'test':
            image_label = now_info['category_id']  # 0-index

        # get nlp
        target = torch.zeros(332).long()
        target[image_label] = 1
        question = now_info['q_token'].long()
        question_label = now_info['q_label']




        return image, image_label, meta, question, target, question_label

