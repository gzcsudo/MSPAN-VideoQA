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

python preprocess/appearance_features.py \
        --gpu_id 0 \
        --dataset msvd-qa \
        --question_type none \
        --feature_type pool5 \
        --num_frames 16

python preprocess/motion_features.py \
        --gpu_id 0 \
        --dataset msvd-qa \
        --question_type none \
        --num_frames 16
