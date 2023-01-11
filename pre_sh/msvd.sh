python3 preprocess/question_features.py \
        --dataset msvd-qa \
        --question_type none \
        --mode total

python3 preprocess/question_features.py \
        --dataset msvd-qa \
        --question_type none \
        --mode train

python3 preprocess/question_features.py \
        --dataset msvd-qa \
        --question_type none \
        --mode val

python3 preprocess/question_features.py \
        --dataset msvd-qa \
        --question_type none \
        --mode test

python3 preprocess/appearance_features.py \
        --gpu_id 0 \
        --dataset msvd-qa \
        --question_type none \
        --feature_type pool5 \
        --num_frames 16

python3 preprocess/motion_features.py \
        --gpu_id 0 \
        --dataset msvd-qa \
        --question_type none \
        --num_frames 16