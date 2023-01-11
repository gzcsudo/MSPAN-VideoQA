python3 preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type transition \
        --mode total

python3 preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type transition \
        --mode train

python3 preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type transition \
        --mode test

python3 preprocess/appearance_features.py \
        --gpu_id 3 \
        --dataset tgif-qa \
        --question_type transition \
        --feature_type pool5 \
        --num_frames 16

python3 preprocess/motion_features.py \
        --gpu_id 3 \
        --dataset tgif-qa \
        --question_type transition \
        --num_frames 16