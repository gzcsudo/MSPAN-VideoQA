python3 train.py \
        --dataset msrvtt-qa \
        --question_type none \
        --T 2 \
        --K 3 \
        --num_scale 8 \
        --num_frames 16 \
        --gpu_id 1 \
        --max_epochs 20 \
        --batch_size 128 \
        --dropout 0.3 \
        --model_id 0 \
        --use_train \
        --use_val \
        --use_test