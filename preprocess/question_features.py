import argparse
import numpy as np
import os
import random
from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt', default='preprocess/pretrained/glove.840.300d.pkl', type=str)
    parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--glove_matrix_pt', type=str, default='data/{}/{}_glove_matrix.pt')
    parser.add_argument('--answers_list', type=str, default='data/{}/{}_answers_list.json')
    parser.add_argument('--answers_count', type=str, default='data/{}/{}_answers_count.json')
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['total', 'train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['action', 'count', 'frameqa', 'transition', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=2020)

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'tgif-qa':
        args.annotation_file = '/home/sudoku/Documents/tgif-qa/csv/{}_{}_question.csv'
        args.output_pt = 'data/tgif-qa/{}/tgif-qa_{}_{}_questions.pt'
        args.glove_matrix_pt = 'data/tgif-qa/{}/tgif-qa_{}_glove_matrix.pt'
        args.vocab_json = 'data/tgif-qa/{}/tgif-qa_{}_vocab.json'
        args.answers_list = 'data/tgif-qa/{}/tgif-qa_{}_answers_list.json'
        args.answers_count = 'data/tgif-qa/{}/tgif-qa_{}_answers_count_{}.json'

        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))
        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)

    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/home/sudoku/Documents/msrvtt-qa/{}_qa.json'.format(args.mode)
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/home/sudoku/Documents/msvd-qa/{}_qa.json'.format(args.mode)
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)