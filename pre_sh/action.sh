python3 preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type action \
        --mode total

python3 preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type action \
        --mode train

python3 preprocess/question_features.py \
        --dataset tgif-qa \
        --question_type action \
        --mode test

python3 preprocess/appearance_features.py \
        --gpu_id 0 \
        --dataset tgif-qa \
        --question_type action \
        --feature_type pool5 \
        --num_frames 16

python3 preprocess/motion_features.py \
        --gpu_id 0 \
        --dataset tgif-qa \
        --question_type action \
        --num_frames 16