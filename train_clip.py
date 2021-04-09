import os
import argparse
import torch
from torch.utils.data import DataLoader
import dataset_RAD
from utils import utils
from clipNet.network import Net

try:
    import _pickle as pickle
except:
    import pickle


def parse_args():

    # Model loading/saving
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='./saved_models/baseline',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
    parser.add_argument('--lr', default=0.005, type=float, metavar='lr',
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='epoch')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')


    # Choices of mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'],
                        help='mode')


    # Question embedding
    parser.add_argument('--question_len', default=77, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--RAD_dir', type=str, default='data2020',
                        help='RAD dir')
    parser.add_argument('--mima', type=bool, default=True,
                        help='MIMA')

    # Network Setting
    parser.add_argument('--lstm_out_size', type=int, default='768',
                        help='lstm')
    parser.add_argument('--word_embedding_size', type=int, default='300',
                        help='word_embedding_dim')
    parser.add_argument('--drop_rate', type=float, default='0.1',
                        help='drop_rate')
    parser.add_argument('--MFB_O', type=int, default='1000',
                        help='MFB_O')
    parser.add_argument('--MFB_K', type=int, default='3',
                        help='MFB_K')
    parser.add_argument('--q_glimse', type=int, default='2',
                        help='glimse_q')
    parser.add_argument('--i_glimse', type=int, default='2',
                        help='glimse_i')
    parser.add_argument('--hidden_size', type=int, default='768',
                        help='if mima 768 else 512')
    parser.add_argument('--HIGH_ORDER', type=bool, default=False,
                        help='high_order')
    parser.add_argument('--activation', type=str, default='relu',
                        help='activation')
    parser.add_argument('--v_dim', type=int, default=128,
                        help='dim of embedding image')
    parser.add_argument('--eps_cnn', type=float, default=1e-5,
                        help='pass')
    parser.add_argument('--momentum_cnn', type=float, default=0.05,
                        help='pass')
    parser.add_argument('--num_stacks', type=int, default=2,
                        help='')

    # coAttention settings
    parser.add_argument('--HIDDEN_SIZE', type=int, default=512,
                        help='')
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=2048,
                        help='')
    parser.add_argument('--LAYER', type=int, default=6,
                        help='')
    parser.add_argument('--DROPOUT_R', type=float, default=0.1,
                        help='')
    parser.add_argument('--FF_SIZE', type=int, default=2048,
                        help='')
    parser.add_argument('--FLAT_MLP_SIZE', type=int, default=512,
                        help='')
    parser.add_argument('--FLAT_GLIMPSES', type=int, default=2,
                        help='')
    parser.add_argument('--FLAT_OUT_SIZE', type=int, default=1024,
                        help='')
    parser.add_argument('--MULTI_HEAD', type=int, default=8,
                        help='')
    parser.add_argument('--HIDDEN_SIZE_HEAD', type=int, default=64,
                        help='')




    args = parser.parse_args()
    return args


# VQA score computation
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    one_hots = one_hots.float()
    labels = labels.float()
    scores = (one_hots * labels)
    return scores

def evaluate(model, args):
    val_set = dataset_RAD.VQAFeatureDataset('val', args, dictionary, question_len=args.question_len)
    val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=0)
    # model.load_state_dict(torch.load(args.output + 'SAN.pth'))
    device = args.device
    score = 0
    upper_bound = 0
    num_data = 0

    with torch.no_grad():
        for i, (v, q, a, _, _) in enumerate(val_loader):
            if args.mima:
                v[0] = v[0].to(device)
                v[1] = v[1].to(device)
            else:
                v = v.to(device)
            q = q.to(device)
            a = a.to(device).float()
            pred = model(v, q)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)
    score = score / len(val_loader.dataset)
    upper_bound = upper_bound / len(val_loader.dataset)
    return score, upper_bound

if __name__ == '__main__':
    args = parse_args()

    # args.MFB_O = 500
    # args.MFB_K = 3
    args.mima = False

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
    train_set = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
    batch_size = args.batch_size

    model = Net(args, len(train_set.label2ans))
    model.to(device)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')


    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.001, [0.9, 0.999])

    # score, upper_bound = evaluate(model, args)


    for epoch in range(args.n_epoch):
        best_score = 0
        best_epoch = 0
        count = 0
        for i, (v, q, a, _, _) in enumerate(train_loader):
            if args.mima:
                v[0] = v[0].to(device)
                v[1] = v[1].to(device)
            else:
                v = v.to(device)
            q = q.to(device)
            a = a.to(device).float()
            pred = model(v, q)
            loss = criterion(pred.float(), a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            print(count)

        """
        score, _ = evaluate(model, args)
        if score > best_score:
            torch.save(model.state_dict(), 'best_model.pth')
            best_epoch = epoch
        print("score ===== " + score)
        print("best_score ======= " + str(best_score) + "  best_epoch ======= " + best_epoch)
        """

        print("epoch===" + str(epoch))






