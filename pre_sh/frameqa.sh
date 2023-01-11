python preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type frameqa \
        --mode total

python preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type frameqa \
        --mode train

python preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type frameqa \
        --mode test

python preprocess/appearance_features.py \
        --gpu_id 2 \
        --dataset tgif-qa \
        --question_type frameqa \
        --feature_type pool5 \
        --num_frames 16

python preprocess/motion_features.py \
        --gpu_id 2 \
        --dataset tgif-qa \
        --question_type frameqa \
        --num_frames 16
