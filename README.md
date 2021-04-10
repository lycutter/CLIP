# environment prepared
1. conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
2. pip install ftfy regex tqdm
3. pip install git+https://github.com/openai/CLIP.git


# code for retraining clip
1. down load the dataset from https://pan.baidu.com/s/1nZC-Z7b7scMLQvVRh5_16w (code: 4t6w), and then put the two pkl file to ./data2020/
2. run python train_clip.py --batch_size xxx
