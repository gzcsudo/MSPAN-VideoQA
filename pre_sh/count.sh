python preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type count \
        --mode total

python preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type count \
        --mode train

python preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type count \
        --mode test

python preprocess/appearance_features.py \
        --gpu_id 1 \
        --dataset tgif-qa \
        --question_type count \
        --feature_type pool5 \
        --num_frames 16

python preprocess/motion_features.py \
        --gpu_id 1 \
        --dataset tgif-qa \
        --question_type count \
        --num_frames 16
