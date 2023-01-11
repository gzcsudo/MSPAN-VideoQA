import argparse
import os
from scipy.misc import imresize
import skvideo.io
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
import torchvision
import random
import numpy as np

from utils import average_sample
from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa


def build_resnet():
    cnn = torchvision.models.resnet152(pretrained=True)
    if args.feature_type == 'pool5':
        model = torch.nn.Sequential(*list(cnn.children())[:-1])
    elif args.feature_type == 'res5c':
        model = torch.nn.Sequential(*list(cnn.children())[:-2])
    elif args.feature_type == 'res4c':
        layers = list(cnn.children())[:-3]
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        model = torch.nn.Sequential(*layers)
    elif args.feature_type == 'pool4':
        layers = list(cnn.children())[:-3]
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        model = torch.nn.Sequential(*layers)
    model = model.cuda()
    model.eval()
    return model


def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    # image_batch = np.asarray(cur_batch)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)
    feats = model(image_batch)
    feats = feats.squeeze()
    feats = feats.data.cpu().clone().numpy()
    return feats


def extract_frames(path):
    valid = True
    try:
        video_data = skvideo.io.vread(path)
    except:
        print('file {} error'.format(path))
        valid = False
        return list(np.zeros(shape=(35, 3, 224, 224))), valid
    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    all_frames = list()
    for frame_data in video_data:
        img = Image.fromarray(frame_data)
        img = imresize(img, img_size, interp='bicubic')
        img = img.transpose(2, 0, 1)[None]
        img = np.array(img)
        all_frames.append(img)
    assert total_frames == len(all_frames)

    return all_frames, valid


def generate_h5(model, video_ids, outfile):
    if args.dataset == "tgif-qa":
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))
    else:
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
    if not os.path.exists(outfile):
        os.makedirs(outfile)

    dataset_size = len(video_ids)
    sample_idx = average_sample(2000, args.num_frames)

    for video_path, video_id in tqdm(video_ids):
        frames, valid = extract_frames(video_path)
        if valid:
            total_frames = len(frames)
            sample_frames = sample_idx[total_frames]
            feats = run_batch(list(map(lambda x: frames[x], sample_frames)), model)

            if args.feature_type == 'pool5':
                assert feats.shape == (args.num_frames, 2048)
            elif args.feature_type == 'res5c':
                assert feats.shape == (args.num_frames, 2048, 7, 7)
            elif args.feature_type == 'res4c':
                assert feats.shape == (args.num_frames, 1024, 7, 7)
            elif args.feature_type == 'pool4':
                assert feats.shape == (args.num_frames, 1024)
        else:
            if args.feature_type == 'pool5':
                feats = np.zeros(shape=(args.num_frames, 2048), dtype=np.float32)
            elif args.feature_type == 'res5c':
                feats = np.zeros(shape=(args.num_frames, 2048, 7, 7), dtype=np.float32)
            elif args.feature_type == 'res4c':
                feats = np.zeros(shape=(args.num_frames, 1024, 7, 7), dtype=np.float32)
            elif args.feature_type == 'pool4':
                feats = np.zeros(shape=(args.num_frames, 1024), dtype=np.float32)

        # print(feats.shape)
        save_npy = os.path.join(outfile, str(video_id) + '.npy')
        np.save(save_npy, feats)

    assert len(os.listdir(outfile)) == dataset_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', default='tgif-qa',
                        choices=['tgif-qa', 'msvd-qa', 'msrvtt-qa'], type=str)
    parser.add_argument('--question_type', default='none',
                        choices=['action', 'count', 'frameqa', 'transition', 'none'], type=str)
    parser.add_argument('--out', dest='outfile',
                        default="data/{}/{}_appearance_{}_{}_feat", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--feature_type', default='pool5',
                        choices=['pool5', 'pool4', 'res5c', 'res4c'], type=str)
    parser.add_argument('--seed', default=2020, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'tgif-qa':
        args.annotation_file = '/home/sudoku/Documents/tgif-qa/csv/Total_{}_question.csv'
        args.video_dir = '/home/sudoku/Documents/tgif-qa/gifs'
        args.outfile = 'data/{}/{}/{}_{}_appearance_{}_{}_feat'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        model = build_resnet()
        generate_h5(model, video_paths,
                    args.outfile.format(args.dataset, args.question_type,
                                        args.dataset, args.question_type, args.feature_type, args.num_frames))
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/home/sudoku/Documents/msrvtt-qa/{}_qa.json'
        args.video_dir = '/home/sudoku/Documents/msrvtt-qa/video/'
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)
        model = build_resnet()
        generate_h5(model, video_paths,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type, args.num_frames))

    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/home/sudoku/Documents/msvd-qa/{}_qa.json'
        args.video_dir = '/home/sudoku/Documents/msvd-qa/video/'
        args.video_name_mapping = '/home/sudoku/Documents/msvd-qa/youtube_mapping.txt'
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        model = build_resnet()
        generate_h5(model, video_paths,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type, args.num_frames))
