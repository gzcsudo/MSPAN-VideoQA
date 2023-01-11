import torch
import os
import pickle
import numpy as np
import random
from torch.utils.data import Dataset


class VideoQADataset(Dataset):
    def __init__(self, features_dir, question_pt, question_type, num_frames):
        super(VideoQADataset, self).__init__()
        self.features_dir = features_dir
        self.question_pt = question_pt
        self.question_type = question_type
        self.num_frames = num_frames

        print('loading features from: \n %s \n %s' % (features_dir[0], features_dir[1]))

        with open(question_pt, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            answers = obj['answers']
            questions_word = obj['questions_word']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        print('loading %d samples from %s' % (len(questions), question_pt))
        self.questions = torch.LongTensor(np.asarray(questions))
        self.questions_len = torch.LongTensor(np.asarray(questions_len))
        self.video_ids = video_ids
        self.answers = answers
        self.questions_word = questions_word

        if self.question_type in ['action', 'transition']:
            self.ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, item):
        question = self.questions[item]
        question_len = self.questions_len[item]
        video_id = self.video_ids[item]
        answer = self.answers[item]
        question_word = self.questions_word[item]

        if self.question_type in ['action', 'transition']:
            ans_candidate = self.ans_candidates[item]
            ans_candidate_len = self.ans_candidates_len[item]
        else:
            ans_candidate = torch.zeros(5)
            ans_candidate_len = torch.zeros(5)

        appearance_pool5_path = os.path.join(self.features_dir[0], str(video_id) + '.npy')
        motion_path = os.path.join(self.features_dir[1], str(video_id) + '.npy')

        app_pool5_feat = np.load(appearance_pool5_path)
        motion_feat = np.load(motion_path)

        assert app_pool5_feat.shape[0] == self.num_frames
        assert app_pool5_feat.shape == motion_feat.shape

        app_pool5_feat = torch.from_numpy(app_pool5_feat)
        motion_feat = torch.from_numpy(motion_feat)

        return (app_pool5_feat, motion_feat, question, question_len,
                ans_candidate, ans_candidate_len, answer, question_word)

    def __len__(self):
        return len(self.questions)
