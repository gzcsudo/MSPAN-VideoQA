import torch
import numpy as np
import argparse
import os
import pickle
import json
from tqdm import tqdm
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from DataLoader import VideoQADataset
from network import VideoQANetwork
from validate import Validate
from utils import *


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def main(args):

    torch.cuda.set_device(args.gpu_id)
    set_seed(args.seed)

    # set data path
    features_dir = []
    if args.dataset == 'tgif-qa':
        for feat_type in args.features_type:
            feat_path = args.features_path.format(
                args.dataset, args.question_type, args.dataset, args.question_type, feat_type)
            features_dir.append(feat_path)
        question_pt_train = args.question_pt.format(
            args.dataset, args.question_type, args.dataset, args.question_type, 'train')
        question_pt_test = args.question_pt.format(
            args.dataset, args.question_type, args.dataset, args.question_type, 'test')
        answers_list_json = args.answers_list_json.format(
            args.dataset, args.question_type, args.dataset, args.question_type)
        glove_matrix_pt = args.glove_matrix_pt.format(
            args.dataset, args.question_type, args.dataset, args.question_type)
        save_dir = args.save_dir.format(args.dataset, args.question_type)
    else:
        for feat_type in args.features_type:
            feat_path = 'data/{}/{}_{}_feat'.format(
                args.dataset, args.dataset, feat_type)
            features_dir.append(feat_path)
        question_pt_train = 'data/{}/{}_{}_questions.pt'.format(
            args.dataset, args.dataset, 'train')
        question_pt_val = 'data/{}/{}_{}_questions.pt'.format(
            args.dataset, args.dataset, 'val')
        question_pt_test = 'data/{}/{}_{}_questions.pt'.format(
            args.dataset, args.dataset, 'test')
        answers_list_json = 'data/{}/{}_answers_list.json'.format(
            args.dataset, args.dataset)
        glove_matrix_pt = 'data/{}/{}_glove_matrix.pt'.format(
            args.dataset, args.dataset)
        save_dir = 'results/exp_{}'.format(args.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.train:
        train_data = VideoQADataset(features_dir, question_pt_train, args.question_type, args.num_frames)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    if args.val:
        val_data = VideoQADataset(features_dir, question_pt_val, args.question_type, args.num_frames)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    if args.test:
        test_data = VideoQADataset(features_dir, question_pt_test, args.question_type, args.num_frames)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)


    if args.question_type in ['action', 'count', 'transition']:
        num_answers = 1
    else:
        print('load answers list')
        with open(answers_list_json, 'r') as f:
            answers_list = json.load(f)
            num_answers = len(answers_list)

    device = torch.device("cuda")
    GCN_adj, GAT_adj = make_adjacency(args)
    print('load glove vectors')
    with open(glove_matrix_pt, 'rb') as f:
        glove_matrix = pickle.load(f)
    glove_matrix = torch.FloatTensor(glove_matrix)
    vocab_size = glove_matrix.shape[0]
    if not args.train:
        model_path = os.path.join(save_dir, 'model_{}.pth'.format(args.model_id))
        args, model_dict = load_model(model_path)
    model_kwargs = {
        'app_pool5_dim': args.app_pool5_dim,
        'motion_dim': args.motion_dim,
        'num_frames': args.num_frames,
        'word_dim': args.word_dim,
        'vocab_size': vocab_size,
        'module_dim': args.module_dim,
        'question_type': args.question_type,
        'num_answers': num_answers,
        'dropout': args.dropout,
        'GCN_adj': GCN_adj,
        'GAT_adj': GAT_adj,
        "K": args.K,
        "T": args.T,
        'num_scale': args.num_scale
    }
    model = VideoQANetwork(**model_kwargs)
    with torch.no_grad():
        model.linguistic.encoder_embed.weight.set_(glove_matrix)
    if not args.train:
        model.load_state_dict(model_dict)
    model = model.to(device)
    if args.question_type == 'count':
        criterion = nn.MSELoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.question_type in ['count', 'none']:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 19, 23, 27], gamma=0.5)
    if not args.train:
        Validate(args, model, test_loader, args.max_epochs, device, val_type="test")
        return 0

    print("Start training........")
    best_acc = 0.0
    best_mse = 1000.0
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_mse = 0.0
        count = 0
        progress_bar = tqdm(train_loader)
        for idx, batch in enumerate(progress_bar):
            input_batch = list(map(lambda x: x.to(device), batch[:-1]))
            optimizer.zero_grad()
            answers = input_batch[-1]
            batch_size = answers.size(0)
            logits = model(*input_batch[: -1])
            if args.question_type in ['action', 'transition']:
                batch_agg = np.concatenate(np.tile(np.arange(batch_size).
                                                   reshape([batch_size, 1]),[1, 5])) * 5
                answers_agg = tile(answers, 0, 5)
                loss = torch.max(torch.tensor(0.0).cuda(),
                                 1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])

                loss = loss.mean()
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (idx + 1)

                optimizer.step()
                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                aggreeings = (preds == answers)

            elif args.question_type == 'count':
                answers = answers.unsqueeze(-1)
                loss = criterion(logits, answers.float())
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (idx + 1)

                optimizer.step()
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2

            else:
                loss = criterion(logits, answers)
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (idx + 1)

                optimizer.step()
                preds = logits.detach().argmax(1)
                aggreeings = (preds == answers)

            if args.question_type == 'count':
                total_mse += batch_mse.sum().item()
                count += batch_size
                progress_bar.set_description("Training epoch \033[1;33m{} \033[0m: loss: \033[1;34m {:.3f} \033[0m, avg_loss: \033[1;35m {:.3f} \033[0m, avg_mse: \033[1;31m {:.4f} \033[0m".format(
                                                    epoch + 1, loss.item(), avg_loss, total_mse / count))
            else:
                total_acc += aggreeings.sum().item()
                count += batch_size
                progress_bar.set_description("Training epoch \033[1;33m{} \033[0m: loss: \033[1;34m {:.3f} \033[0m, avg_loss: \033[1;35m {:.3f} \033[0m, avg_acc: \033[1;31m {:.4f} \033[0m".format(
                                                    epoch + 1, loss.item(), avg_loss, total_acc / count))
        scheduler.step()
        progress_bar.close()

        if args.val:
            if args.question_type == 'count':
                val_mse = Validate(args, model, val_loader, epoch, device, val_type="val")
            else:
                if args.dataset in ['msvd-qa', 'msrvtt-qa']:
                    epoch_accuracy_word, val_acc = Validate(args, model, val_loader, epoch, device, val_type="val")
                else:
                    val_acc = Validate(args, model, val_loader, epoch, device, val_type="val")

        if args.test:
            if args.question_type == 'count':
                test_mse = Validate(args, model, test_loader, epoch, device, val_type="test")
                if test_mse < best_mse:
                    best_mse = test_mse
                    save_model(args, model, os.path.join(save_dir, 'model_{}.pth'.format(args.model_id)), best_mse)
            else:
                if args.dataset in ['msvd-qa', 'msrvtt-qa']:
                    epoch_accuracy_word, test_acc = Validate(args, model, test_loader, epoch, device, val_type="test")
                else:
                    test_acc = Validate(args, model, test_loader, epoch, device, val_type="test")
                if test_acc > best_acc:
                    best_acc = test_acc
                    accuracy_word = None
                    if args.dataset in ['msvd-qa', 'msrvtt-qa']:
                        accuracy_word = epoch_accuracy_word
                    save_model(args, model, os.path.join(save_dir, 'model_{}.pth'.format(args.model_id)), best_acc, accuracy_word)

    print('~~~~~Finall best results !~~~~~~~~~')
    if args.question_type == 'count':
        print('Best MSE on {}_{} is: \033[1;31m {:.4f} \033[0m'.format(args.dataset, args.question_type, best_mse))
    else:
        if args.dataset in ['msvd-qa', 'msrvtt-qa']:
            print('Best Question Words Accuracy: what {:.4f}, who {:.4f}, how {:.4f}, when {:.4f}, where {:.4f}'.format(
                    accuracy_word['what'], accuracy_word['who'], accuracy_word['how'], accuracy_word['when'], accuracy_word['where']))
            print('Best Accuracy on {} is: \033[1;31m {:.4f} \033[0m'.format(args.dataset, best_acc))
        else:
            print('Best Accuracy on {}_{} is: \033[1;31m {:.4f} \033[0m'.format(args.dataset, args.question_type, best_acc))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', default='data/{}/{}/{}_{}_{}_feat', type=str)
    parser.add_argument('--question_pt', type=str, default='data/{}/{}/{}_{}_{}_questions.pt')
    parser.add_argument('--glove_matrix_pt', type=str, default='data/{}/{}/{}_{}_glove_matrix.pt')
    parser.add_argument('--answers_list_json', type=str, default='data/{}/{}/{}_{}_answers_list.json')
    parser.add_argument('--save_dir', default='results/exp_{}_{}', type=str)
    parser.add_argument('--dataset', default='tgif-qa',
                        choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--question_type', default='none',
                        choices=['action', 'count', 'frameqa', 'transition', 'none'], type=str)
    parser.add_argument('--features_type',
                        default=['appearance_pool5_16', 'motion_16'], type=str)

    # hyper-parameters
    parser.add_argument('--num_scale', default=6, type=int)
    parser.add_argument('--T', default=2, type=int)
    parser.add_argument('--K', default=3, type=int)

    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--module_dim', default=512, type=int)
    parser.add_argument('--app_pool5_dim', default=2048, type=int)
    parser.add_argument('--motion_dim', default=2048, type=int)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--max_epochs', default=25, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.000005, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--model_id', default=0, type=int)

    parser.add_argument('--use_train', default=False, dest='train', action='store_true')
    parser.add_argument('--use_val', default=False, dest='val', action='store_true')
    parser.add_argument('--use_test', default=False, dest='test', action='store_true')

    args = parser.parse_args()
    main(args)