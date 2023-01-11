python3 preprocess/question_features.py \
        --dataset msrvtt-qa \
        --question_type none \
        --mode total

python3 preprocess/question_features.py \
        --dataset msrvtt-qa \
        --question_type none \
        --mode train

python3 preprocess/question_features.py \
        --dataset msrvtt-qa \
        --question_type none \
        --mode val

python3 preprocess/question_features.py \
        --dataset msrvtt-qa \
        --question_type none \
        --mode test

python3 preprocess/appearance_features.py \
        --gpu_id 1 \
        --dataset msrvtt-qa \
        --question_type none \
        --feature_type pool5 \
        --num_frames 16

python3 preprocess/motion_features.py \
        --gpu_id 1 \
        --dataset msrvtt-qa \
        --question_type none \
        --num_frames 16