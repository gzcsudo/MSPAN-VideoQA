import argparse
import os
from scipy.misc import imresize
import skvideo.io
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import random
import numpy as np

from utils import average_sample
from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
from resnet import generate_model


def build_resnet(resnet_depth=152, n_classes=700, pretrained_path=None):
    model = generate_model(resnet_depth, n_classes=n_classes, use_fc=False)
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists(pretrained_path)
    model_data = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    return model

def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.224])

    # image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = np.asarray(cur_batch)
    # print(image_batch.shape)
    image_batch = image_batch.transpose(0, 1, 3, 4, 2)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = image_batch.transpose(0, 4, 1, 2, 3)
    # print(image_batch.shape)
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)
    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()
    return feats


def extract_frames(path, num_frames_per_clip=16):
    valid = True
    try:
        video_data = skvideo.io.vread(path)
    except:
        print('file {} error'.format(path))
        valid = False
        return list(np.zeros(shape=(35, 16, 3, 112, 112))), valid
    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    all_frames = list()
    for i in range(total_frames):
        img = Image.fromarray(video_data[i])
        img = imresize(img, img_size, interp='bicubic')
        img = img.transpose(2, 0, 1)[None]
        img = np.array(img)
        all_frames.append(img)
    assert len(all_frames) == total_frames
    target_frames = list()
    for i in range(total_frames):
        temp_frames = []
        for j in range(-num_frames_per_clip//2, num_frames_per_clip//2):
            pos = min(max(i + j, 0), total_frames-1)
            temp_frames.append(all_frames[pos])
        temp_frames = np.concatenate(temp_frames, axis=0)
        # print(temp_frames.shape)
        target_frames.append(temp_frames)
    assert len(target_frames) == total_frames
    return target_frames, valid



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
        else:
            feats = np.zeros(shape=(args.num_frames, 2048), dtype=np.float32)
        assert feats.shape == (args.num_frames, 2048)
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
                        default="data/{}/{}_motion_{}_feat", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_height', default=112, type=int)
    parser.add_argument('--image_width', default=112, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--resnet_depth', default=152,
                        choices=[10, 18, 34, 50, 101, 152, 200], type=int)
    parser.add_argument('--n_classes', default=700,
                        choices=[700, 1039], type=int)
    parser.add_argument('--pretrained_path', default='preprocess/pretrained/r3d152_K_200ep.pth',
                        choices=['preprocess/pretrained/r3d101_K_200ep.pth',
                                 'preprocess/pretrained/r3d152_K_200ep.pth',
                                 'preprocess/pretrained/r3d200_K_200ep.pth'], type=str)
    parser.add_argument('--seed', default=2020, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'tgif-qa':
        args.annotation_file = '/home/sudoku/Documents/tgif-qa/csv/Total_{}_question.csv'
        args.video_dir = '/home/sudoku/Documents/tgif-qa/gifs'
        args.outfile = 'data/{}/{}/{}_{}_motion_{}_feat'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        model = build_resnet(args.resnet_depth, args.n_classes, args.pretrained_path)
        generate_h5(model, video_paths,
                    args.outfile.format(args.dataset, args.question_type,
                                        args.dataset, args.question_type, args.num_frames))
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/home/sudoku/Documents/msrvtt-qa/{}_qa.json'
        args.video_dir = '/home/sudoku/Documents/msrvtt-qa/video/'
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)
        model = build_resnet(args.resnet_depth, args.n_classes, args.pretrained_path)
        generate_h5(model, video_paths,
                    args.outfile.format(args.dataset, args.dataset, args.num_frames))
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/home/sudoku/Documents/msvd-qa/{}_qa.json'
        args.video_dir = '/home/sudoku/Documents/msvd-qa/video/'
        args.video_name_mapping = '/home/sudoku/Documents/msvd-qa/youtube_mapping.txt'
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        model = build_resnet(args.resnet_depth, args.n_classes, args.pretrained_path)
        generate_h5(model, video_paths,
                    args.outfile.format(args.dataset, args.dataset, args.num_frames))
