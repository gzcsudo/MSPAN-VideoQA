python3 train.py \
        --dataset msvd-qa \
        --question_type none \
        --T 2 \
        --K 3 \
        --num_scale 8 \
        --num_frames 16 \
        --gpu_id 0 \
        --max_epochs 30 \
        --batch_size 128 \
        --dropout 0.3 \
        --model_id 0 \
        --use_test \
        --use_val \
        --use_train