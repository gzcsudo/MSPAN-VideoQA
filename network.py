import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import random
from SSP import *


class Linguistic_embedding(nn.Module):
    def __init__(self, vocab_size, word_dim=300, module_dim=512, dropout=0.2):
        super(Linguistic_embedding, self).__init__()

        self.encoder_embed = nn.Embedding(vocab_size, word_dim)
        self.encoder = nn.LSTM(word_dim, module_dim // 2, num_layers=3, batch_first=True, bidirectional=True)
        self.dropout_global = nn.Dropout(dropout)

    def forward(self, questions, question_len):
        questions_embedding = self.encoder_embed(questions)
        embed = nn.utils.rnn.pack_padded_sequence(questions_embedding, question_len,
                                                  batch_first=True, enforce_sorted=False)
        self.encoder.flatten_parameters()
        _, (question_global, _) = self.encoder(embed)
        question_global = torch.cat([question_global[0], question_global[1]], dim=-1)
        question_global = self.dropout_global(question_global)

        return question_global


class VideoQANetwork(nn.Module):
    def __init__(self, **kwargs):
        super(VideoQANetwork, self).__init__()
        self.app_pool5_dim = kwargs.pop("app_pool5_dim")
        self.motion_dim = kwargs.pop("motion_dim")
        self.num_frames = kwargs.pop("num_frames")
        self.word_dim = kwargs.pop("word_dim")
        self.vocab_size = kwargs.pop("vocab_size")
        self.module_dim = kwargs.pop("module_dim")
        self.question_type = kwargs.pop("question_type")
        self.num_answers = kwargs.pop("num_answers")
        self.num_scale = kwargs.pop("num_scale")
        self.dropout = kwargs.pop("dropout")

        self.GCN_adj = kwargs.pop("GCN_adj")
        self.GAT_adj = kwargs.pop("GAT_adj")
        self.K = kwargs.pop("K")
        self.T = kwargs.pop("T")

        self.agg_pools = nn.Sequential(*[nn.MaxPool1d(kernel_size=i+1, stride=1) for i in range(self.num_scale)])
        self.linguistic = Linguistic_embedding(self.vocab_size, self.word_dim, self.module_dim, self.dropout)
        self.visual_dim = self.app_pool5_dim

        self.ssp_offer = MSPAN(self.visual_dim, self.module_dim, self.num_frames, self.num_scale,
                               self.GCN_adj, self.GAT_adj, self.T, self.K, self.dropout)

        self.fc_cat = nn.Linear(self.visual_dim * 2, self.visual_dim)
        self.que_visual = Fusion(self.visual_dim, self.module_dim, use_bias=True, dropout=self.dropout)

        if self.question_type in ['action', 'transition']:
            self.ans_visual = Fusion(self.visual_dim, self.module_dim, use_bias=True, dropout=self.dropout)
            self.fc_que = nn.Linear(self.module_dim, self.module_dim, bias=False)
            self.fc_ans = nn.Linear(self.module_dim, self.module_dim, bias=False)
            self.classifier = nn.Sequential(
                nn.Linear(self.module_dim * 4, self.module_dim),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.module_dim, self.num_answers)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.module_dim, self.module_dim),
                nn.ELU(),
                nn.BatchNorm1d(self.module_dim),
                nn.Dropout(self.dropout),
                nn.Linear(self.module_dim, self.num_answers)
            )

    def forward(self, app_pool5_feat, motion_feat, question, question_len, ans_candidate, ans_candidate_len):
        B = question.size(0)

        question_feat = self.linguistic(question, question_len)
        visual_feat = torch.cat([app_pool5_feat, motion_feat], dim=-1)
        visual_feat = self.fc_cat(visual_feat).permute(0, 2, 1)
        graphs = list(map(lambda pool: pool(visual_feat).permute(0, 2, 1), self.agg_pools))

        visual_feat = self.ssp_offer(graphs, question_feat)
        que_feat = self.que_visual(visual_feat, question_feat)

        if self.question_type in ['action', 'transition']:
            candidate_agg = ans_candidate.reshape(-1, ans_candidate.size(2))
            candidate_len_agg = ans_candidate_len.reshape(-1)
            batch_agg = np.reshape(np.tile(np.expand_dims(np.arange(B), axis=1), [1, 5]), [-1])
            candidate_global = self.linguistic(candidate_agg, candidate_len_agg)

            ans_feat = self.ans_visual(visual_feat[batch_agg], candidate_global)
            que_feat = torch.cat([que_feat[batch_agg], ans_feat,
                                  self.fc_que(question_feat)[batch_agg], self.fc_ans(candidate_global)], dim=-1)

        logits = self.classifier(que_feat)

        return logits