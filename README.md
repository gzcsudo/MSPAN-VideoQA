# MSPAN-VideoQA

[Multi-Scale Progressive Attention Network for Video Question Answering](https://aclanthology.org/2021.acl-short.122/), 
[ACL 2021](https://2021.aclweb.org/).

Zhicheng Guo, Jiaxuan Zhao, Licheng Jiao, Xu Liu, Lingling Li

## Setups

1. Install the python dependency packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Download [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [MSVD-QA, MSRVTT-QA](https://github.com/xudejing/video-question-answering) datasets and edit absolute paths in `preprocess/question_features.py` , `preprocess/appearance_features.py` and `preprocess/motion_features.py` upon where you locate your data.

## Preprocessing features

For above three datasets of VideoQA, you can choose 3 options of `--dataset`: 

`tgif-qa`, `msvd-qa` and `msrvtt-qa`.

For different datasets, you can choose 5 options of `--question_type`: 

`none`, `action`, `count`, `frameqa` and `transition`.

### Extracting question features

1. Download [Glove 300D](http://nlp.stanford.edu/data/glove.840B.300d.zip) to `preprocess/pretrained/` and process it into a pickle file:

   ```bash
   python preprocess/txt2pickle.py
   ```

2. To extract question features.

   For TGIF-QA dataset:

   ```bash
   python preprocess/question_features.py 
           --dataset tgif-qa \
           --question_type action \
           --mode total
   python preprocess/question_features.py \
           --dataset tgif-qa \
           --question_type action \
           --mode train
   python preprocess/question_features.py \
           --dataset tgif-qa \
           --question_type action \
           --mode test
   ```
   
   For MSVD-QA/MSRVTT-QA dataset:
   
   ```bash
   python preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode total
   python preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode train
   python preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode val
   python preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode test
   ```

### Extracting visual features

 1. Download pre-trained [3D-ResNet152](https://drive.google.com/file/d/1U7p9iIgkZviuKvpObzN6gx5OiflmAKaT/view?usp=sharing) to `preprocess/pretrained/` .

    You can learn more about this model in the following paper:

    ["Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs", arXiv preprint, 2020.](https://arxiv.org/abs/2004.04968)

2. To extract appearance features:

   ```bash
   python preprocess/appearance_features.py \
           --gpu_id 0 \
           --dataset tgif-qa \
           --question_type action \
           --feature_type pool5 \
           --num_frames 16
   ```

3. To extract motion features:

   ```bash
   python preprocess/motion_features.py \
           --gpu_id 0 \
           --dataset tgif-qa \
           --question_type action \
           --num_frames 16
   ```

## Training

You can choose the suitable `--dataset` and `--question_type` to start training:

```bash
python train.py \
        --dataset tgif-qa \
        --question_type action \
        --T 2 \
        --K 3 \
        --num_scale 8 \
        --num_frames 16 \
        --gpu_id 0 \
        --max_epochs 30 \
        --batch_size 64 \
        --dropout 0.1 \
        --model_id 0 \
        --use_test \
        --use_train
```

Or, you can run the following command to start training:

```bash
sh train_sh/action.sh
```

You can see the training commands for all datasets and tasks under the `train_sh` folder.

#### Evaluation

To evaluate the trained model, run the following command:

```bash
sh test_sh/action.sh
```

You can see the evaluating commands for all datasets and tasks under the `test_sh` folder.

## Citation

```
@inproceedings{guo2021multi,
  title={Multi-scale progressive attention network for video question answering},
  author={Guo, Zhicheng and Zhao, Jiaxuan and Jiao, Licheng and Liu, Xu and Li, Lingling},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
  pages={973--978},
  year={2021}
}
```
