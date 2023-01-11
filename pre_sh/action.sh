python preprocess/question_features.py \
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

python preprocess/appearance_features.py \
        --gpu_id 0 \
        --dataset tgif-qa \
        --question_type action \
        --feature_type pool5 \
        --num_frames 16

python preprocess/motion_features.py \
        --gpu_id 0 \
        --dataset tgif-qa \
        --question_type action \
        --num_frames 16
