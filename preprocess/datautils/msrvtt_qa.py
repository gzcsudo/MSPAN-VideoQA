import json
import nltk
import pickle
import numpy as np

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


def load_video_paths(args):
    video_paths = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        with open(args.annotation_file.format(mode), 'r') as anno_file:
            instances = json.load(anno_file)
        video_ids = [instance['video_id'] for instance in instances]
        video_ids = set(video_ids)
        if mode in ['train', 'val']:
            for video_id in video_ids:
                video_paths.append((args.video_dir + 'TrainValVideo/video{}.mp4'.format(video_id), video_id))
        else:
            for video_id in video_ids:
                video_paths.append((args.video_dir + 'TestVideo/video{}.mp4'.format(video_id), video_id))

    return video_paths


def process_questions(args):
    print('Loading data')

    if args.mode == 'total':
        all_instances = []
        for mode in ['train', 'val', 'test']:
            with open(args.annotation_file.replace('total',mode), 'r') as dataset_file:
                instances = json.load(dataset_file)
                all_instances.extend(instances)

        print("Total number of questions:", len(all_instances))
        print('Building vocab')
        token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, instance in enumerate(all_instances):
            question = instance['question'].lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in token_to_idx:
                    token_to_idx[token] = len(token_to_idx)
        print('Get token_to_idx, num: %d' % len(token_to_idx))
        vocab = token_to_idx
        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
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
        print('Write into %s' % args.glove_matrix_pt.format(args.dataset, args.dataset))
        with open(args.glove_matrix_pt.format(args.dataset, args.dataset), 'wb') as f:
            pickle.dump(glove_matrix, f)

        all_instances = []
        for mode in ['train', 'val']:
            with open(args.annotation_file.replace('total', mode), 'r') as dataset_file:
                instances = json.load(dataset_file)
                all_instances.extend(instances)

        answers_cnt = {}
        for instance in all_instances:
            answer = instance['answer']
            answers_cnt[answer] = answers_cnt.get(answer, 0) + 1

        print("Number of all answers:", len(answers_cnt))

        sort_answers_cnt = []
        for token in answers_cnt:
            sort_answers_cnt.append((answers_cnt[token], token))
        sort_answers_cnt.sort(reverse=True)
        # print(sort_answers_cnt)
        answers_count = {}
        for ans in sort_answers_cnt:
            answers_count[ans[1]] = ans[0]

        print('Write into %s' % args.answers_count.format(args.dataset, args.dataset))
        with open(args.answers_count.format(args.dataset, args.dataset), 'w') as f:
            json.dump(answers_count, f, indent=4)

        answers_list = {'<UNK0>': 0, '<UNK1>': 1, '<UNK2>': 2}
        for i in range(len(sort_answers_cnt)):
            if len(answers_list) == args.answer_top:
                break
            answer = sort_answers_cnt[i][1]
            if not answer in answers_list:
                answers_list[answer] = len(answers_list)

        print("Number of unique answers:", len(answers_list))
        print('Write into %s' % args.answers_list.format(args.dataset, args.dataset))
        with open(args.answers_list.format(args.dataset, args.dataset), 'w') as f:
            json.dump(answers_list, f, indent=4)

    else:

        print('Encoding data')
        print("Load vocab from %s" % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)
        print('Load answers_list from %s' % args.answers_list.format(args.dataset, args.dataset))
        with open(args.answers_list.format(args.dataset, args.dataset), 'r') as f:
            answers_list = json.load(f)
        print('Load instances from %s' % args.annotation_file)
        with open(args.annotation_file, 'r') as dataset_file:
            instances = json.load(dataset_file)

        questions_encoded = []
        questions_len = []
        question_ids = []
        all_answers = []
        video_ids = []
        video_names = []
        questions_word = []
        for idx, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            question_tokens = nltk.word_tokenize(question)
            questions_word.append(question_tokens[0])
            question_encoded = encode(question_tokens, vocab, allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_ids.append(idx)

            im_name = instance['video_id']
            video_ids.append(im_name)
            video_names.append(im_name)

            ans = instance['answer']
            if ans in answers_list:
                ans_id = answers_list[ans]
            else:
                if args.mode == 'train':
                    ans_id = answers_list['<UNK0>']
                elif args.mode == 'val':
                    ans_id = answers_list['<UNK1>']
                else:
                    ans_id = answers_list['<UNK2>']
            # print(ans_id)
            all_answers.append(ans_id)

        max_question_length = max(x for x in questions_len)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['<NULL>'])

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
        obj = {
            'questions': questions_encoded,
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids),
            'video_names': np.array(video_names),
            'answers': all_answers,
            'questions_word': questions_word
        }
        with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)
