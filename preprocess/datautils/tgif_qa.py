import os
import pandas as pd
import json
import nltk
import pickle
from tqdm import tqdm
import numpy as np


def load_video_paths(args):
    input_paths = []
    annotation = pd.read_csv(args.annotation_file.format(args.question_type), delimiter='\t')
    gif_names = list(annotation['gif_name'])
    keys = list(annotation['key'])
    print("Number of questions: {}".format(len(gif_names)))
    for idx, gif in enumerate(gif_names):
        gif_abs_path = os.path.join(args.video_dir, ''.join([gif, '.gif']))
        input_paths.append((gif_abs_path, keys[idx]))
    input_paths = list(set(input_paths))
    print("Number of unique videos: {}".format(len(input_paths)))

    return input_paths


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates):
    questions_encoded = []
    ans_candidates_encoded = []
    questions_len = []
    ans_candidates_len = []
    question_ids = []
    questions_word = []
    correct_answers = []
    answers_cnt = {}

    for idx, question in enumerate(tqdm(questions)):
        question_ids.append(idx)
        correct_answers.append(int(answers[idx]))
        # encode question
        question = question.replace('-',' ')
        question = question.replace('/', ' ')
        question_words = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question_words)
        questions_word.append(question_tokens[0])
        question_encoded = encode(question_tokens, vocab, allow_unk=True)
        questions_encoded.append(question_encoded)
        que_len = len(question_encoded)
        questions_len.append(que_len)
        # encode answer
        candidates_encoded = []
        candidates_len = []
        for ans in ans_candidates[idx]:
            ans_words = ans.lower()
            ans_tokens = nltk.word_tokenize(ans_words)
            cand_encoded = encode(ans_tokens, vocab, allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))
        ans_candidates_encoded.append(candidates_encoded)
        ans_candidates_len.append(candidates_len)
        # count answer
        ans_id = int(answers[idx])
        ans = ans_candidates[idx][ans_id]
        ans = ans.lower()
        answers_cnt[ans] = answers_cnt.get(ans, 0) + 1

    sort_answers_cnt = []
    for token in answers_cnt:
        sort_answers_cnt.append((answers_cnt[token], token))
    sort_answers_cnt.sort(reverse=True)
    # print(sort_answers_cnt)
    answers_count = {}
    for ans in sort_answers_cnt:
        answers_count[ans[1]] = ans[0]

    answers_count_path = args.answers_count.format(args.question_type, args.question_type, args.mode)
    print('Write into %s' % answers_count_path)
    with open(answers_count_path, 'w') as f:
        json.dump(answers_count, f, indent=4)

    # Pad encoded questions
    max_questions_length = max(questions_len)
    for qe in questions_encoded:
        while len(qe) < max_questions_length:
            qe.append(vocab['<NULL>'])
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded answers
    max_answer_cand_length = max(max(candidate) for candidate in ans_candidates_len)
    for ans_cands in ans_candidates_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_cand_length:
                ans.append(vocab['<NULL>'])
    ans_candidates_encoded = np.asarray(ans_candidates_encoded, dtype=np.int32)
    ans_candidates_len = np.asarray(ans_candidates_len, dtype=np.int32)
    print(questions_encoded.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, args.mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids),
        'video_names': np.array(video_names),
        'ans_candidates': ans_candidates_encoded,
        'ans_candidates_len': ans_candidates_len,
        'answers': correct_answers,
        'questions_word': questions_word
    }
    with open(args.output_pt.format(args.question_type, args.question_type, args.mode), 'wb') as f:
        pickle.dump(obj, f)


def process_questions_mulchoices(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    elif args.mode in ["test"]:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Total", args.question_type), delimiter='\t')

    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    ans_candidates = ans_candidates.transpose()
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk

    print('~~~~~show a sample of dataset~~~~~~')
    print('video_name: {}'.format(video_names[0]))
    print('video_id: {}'.format(video_ids[0]))
    print('question: {}'.format(questions[0]))
    print('ans_candidate: {}'.format(ans_candidates[0]))
    print('answer: {}'.format(answers[0]))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if args.mode in ['total']:
        print('Building vocab')
        token_to_idx = {'<NULL>': 0, '<UNK>': 1}

        # answers token
        for candidates in ans_candidates:
            for ans in candidates:
                ans = ans.lower()
                for token in nltk.word_tokenize(ans):
                    if token not in token_to_idx:
                        token_to_idx[token] = len(token_to_idx)
        # questions token
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in token_to_idx:
                    token_to_idx[token] = len(token_to_idx)


        print('Get total token_to_idx: {}'.format(len(token_to_idx)))
        vocab = token_to_idx
        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for word in vocab:
            vector = glove.get(word, np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)
        print('Write into %s' % args.glove_matrix_pt.format(args.question_type, args.question_type))
        with open(args.glove_matrix_pt.format(args.question_type, args.question_type), 'wb') as f:
            pickle.dump(glove_matrix, f)

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates)


def openended_encoding_data(args, vocab, questions, video_names, video_ids, answers):
    questions_encoded = []
    questions_len = []
    question_ids = []
    questions_word = []
    correct_answers = []
    answers_cnt = {}
    answers_to_idx = {'<UNK>': 0}

    for idx, question in enumerate(tqdm(questions)):
        question_ids.append(idx)
        # encode question
        question = question.replace('-', ' ')
        question = question.replace('/', ' ')
        question_words = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question_words)
        questions_word.append(question_tokens[0])
        question_encoded = encode(question_tokens, vocab, allow_unk=True)
        questions_encoded.append(question_encoded)
        que_len = len(question_encoded)
        questions_len.append(que_len)

        # count answer
        if args.question_type == 'frameqa':
            answer = answers[idx]
            if answer not in answers_to_idx:
                answers_to_idx[answer] = len(answers_to_idx)
            answers_cnt[answer] = answers_cnt.get(answer, 0) + 1


    if args.question_type in ['frameqa']:

        sort_answers_cnt = []
        for token in answers_cnt:
            sort_answers_cnt.append((answers_cnt[token], token))
        sort_answers_cnt.sort(reverse=True)
        # print(sort_answers_cnt)
        answers_count = {}
        for ans in sort_answers_cnt:
            answers_count[ans[1]] = ans[0]

        answers_count_path = args.answers_count.format(args.question_type, args.question_type, args.mode)
        print('Write into %s' % answers_count_path)
        with open(answers_count_path, 'w') as f:
            json.dump(answers_count, f, indent=4)

        if args.mode == 'train':
            print('Write into %s' % args.answers_list.format(args.question_type, args.question_type))
            with open(args.answers_list.format(args.question_type, args.question_type), 'w') as f:
                json.dump(answers_to_idx, f, indent=4)
        else:
            print('Load from %s' % args.answers_list.format(args.question_type, args.question_type))
            with open(args.answers_list.format(args.question_type, args.question_type), 'r') as f:
                answers_to_idx = json.load(f)

    # print(answers_to_idx, len(answers_to_idx))
    unk_answers = []
    for idx, ans in enumerate(answers):
        if args.question_type in ['count']:
            correct_answers.append(max(int(ans), 1))
        else:
            if ans in answers_to_idx:
                ans_id = answers_to_idx[ans]
            else:
                ans_id = answers_to_idx['<UNK>']
                unk_answers.append(ans)
            correct_answers.append(ans_id)

    print('unknow answer number: {}'.format(len(unk_answers)))
    # print(len(correct_answers), correct_answers)

    # Pad encoded questions
    max_questions_length = max(questions_len)
    for qe in questions_encoded:
        while len(qe) < max_questions_length:
            qe.append(vocab['<NULL>'])
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, args.mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids),
        'video_names': np.array(video_names),
        'answers': correct_answers,
        'questions_word': questions_word
    }
    with open(args.output_pt.format(args.question_type, args.question_type, args.mode), 'wb') as f:
        pickle.dump(obj, f)


def process_questions_openended(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    elif args.mode in ["test"]:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Total", args.question_type), delimiter='\t')

    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])

    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk

    print('~~~~~show a sample of dataset~~~~~~')
    print('video_name: {}'.format(video_names[0]))
    print('video_id: {}'.format(video_ids[0]))
    print('question: {}'.format(questions[0]))
    print('answer: {}'.format(answers[0]))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if args.mode in ['total']:
        print('Building vocab')
        token_to_idx = {'<NULL>': 0, '<UNK>': 1}

        # questions and objects token
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in token_to_idx:
                    token_to_idx[token] = len(token_to_idx)

        print('Get total token_to_idx: {}'.format(len(token_to_idx)))
        vocab = token_to_idx
        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for word in vocab:
            vector = glove.get(word, np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)
        print('Write into %s' % args.glove_matrix_pt.format(args.question_type, args.question_type))
        with open(args.glove_matrix_pt.format(args.question_type, args.question_type), 'wb') as f:
            pickle.dump(glove_matrix, f)

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        openended_encoding_data(args, vocab, questions, video_names, video_ids, answers)