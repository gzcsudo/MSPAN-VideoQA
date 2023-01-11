python3 train.py \
        --dataset tgif-qa \
        --question_type frameqa \
        --T 2 \
        --K 3 \
        --num_scale 8 \
        --num_frames 16 \
        --gpu_id 2 \
        --max_epochs 30 \
        --batch_size 64 \
        --dropout 0.3 \
        --model_id 0 \
        --use_test \
        --use_train