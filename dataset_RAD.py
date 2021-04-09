"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
from utils import utils
import torch
from torch.utils.data import Dataset
import itertools
import warnings
import en_core_web_md
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
COUNTING_ONLY = False
# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('image_name')
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'    : data['image_name'],
        'image'       : img,
        'question'    : data['question'],
        'answer'      : answer,
        'answer_type' : data['answer_type'],
        'question_type': data['question_type']
    }
    return entry

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError:
    return False
  return True

def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])

    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])

    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries

class VQAFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data', question_len=12):
        super(VQAFeatureDataset, self).__init__()
        self.args = args
        assert name in ['train', 'val']
        dataroot = args.RAD_dir
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        # End get the number of answer type class
        self.dictionary = dictionary

        # TODO: load img_id2idx
        imgid2idx_file = 'imgid2idx_' + name + '.json'
        self.img_id2idx = json.load(open(os.path.join(dataroot, imgid2idx_file)))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)

        # load image data
        if args.mima == False:
            img_files = name + '_img_224x224.pkl'
            images_path = os.path.join(dataroot, img_files)
            self.img_data = cPickle.load(open(images_path, 'rb'))
        else:
            self.img_data = []
            path1 = name + '_img_84x84.pkl'
            path2 = name + '_img_128x128.pkl'
            img_data1 = os.path.join(dataroot, path1)
            img_data2 = os.path.join(dataroot, path2)
            self.img_data.append(cPickle.load(open(img_data1, 'rb')))
            self.img_data.append(cPickle.load(open(img_data2, 'rb')))


        transform_list = [ToTensor()]
        self.transform = transforms.Compose(transform_list)

        # tokenization
        self.tokenize(question_len)
        self.tensorize()

        if args.mima == False:
            self.pretrain_emb = self.embbeding()


    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):

        # self.img_data = torch.from_numpy(self.img_data)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token'])).long()
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def embbeding(self):
        pretrained_emb = []
        spacy_tool = en_core_web_md.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool('CLS').vector)
        for i in range(len(self.dictionary.idx2word)):
            word = self.dictionary.idx2word[i]
            pretrained_emb.append(spacy_tool(word).vector)
        return np.array(pretrained_emb)

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
        if self.args.mima:
            image_data = []
            image_data.append(self.img_data[0][entry['image']])
            image_data.append(self.img_data[1][entry['image']])
            image_data[0] = self.transform(image_data[0])
            image_data[1] = self.transform(image_data[1])
        else:
            image_data = self.img_data[entry['image']]
            image_data = self.transform(image_data)

        if None!=answer:
            labels = answer['labels'].long()
            scores = answer['scores'].long()
            target = torch.zeros(self.num_ans_candidates).long()
            if labels is not None:
                target.scatter_(0, labels, scores)
            return image_data, question, target, answer_type, question_type

        else:
            return image_data, question, answer_type, question_type

    def __len__(self):
        return len(self.entries)

def tfidf_from_questions(names, args, dictionary, dataroot='data', target=['rad']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    if args.use_RAD:
        dataroot = args.RAD_dir
    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'rad' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + 'set.json')
            questions = json.load(open(question_path))
            for question in questions:
                populate(inds, df, question['question'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = os.path.join(dataroot, 'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights

if __name__=='__main__':
    # dictionary = Dictionary.load_from_file('data_RAD/dictionary.pkl')
    # tfidf, weights = tfidf_from_questions(['train'], None, dictionary)
    # w_emb = WordEmbedding(dictionary.ntoken, 300, .0, 'c')
    # w_emb.init_embedding(os.path.join('data_RAD', 'glove6b_init_300d.npy'), tfidf, weights)
    # with open('data_RAD/embed_tfidf_weights.pkl', 'wb') as f:
    #     torch.save(w_emb, f)
    # print("Saving embedding with tfidf and weights successfully")

    dictionary = Dictionary.load_from_file('data_RAD/dictionary.pkl')
    w_emb = WordEmbedding(dictionary.ntoken, 300, .0, 'c')
    with open('data_RAD/embed_tfidf_weights.pkl', 'rb') as f:
        w_emb = torch.load(f)
    print("Load embedding with tfidf and weights successfully")
