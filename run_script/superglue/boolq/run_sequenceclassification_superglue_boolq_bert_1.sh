# source activate
# conda activate py39

export TASK_NAME=superglue
# export DATASET_NAME=boolq
export CUDA_VISIBLE_DEVICES=7
# export MODEL_NAME=bert_base_uncased
# export MODEL_DIR=/opt/meituan/cephfs_caizefan/models/
# export MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/caizefan/models/torch/
export MODEL_DIR=/home/caizf/models/


# 大模型（1024）要用小学习率，1e-5，5e-6
# 小模型（768）用大学习率 2e-5
# 大模型参数锁定，prompt网络更新用大学习率，1e-2
# 16G显卡 128 max_seq_length 大模型（1024）要用bs
# 16G显卡 128 max_seq_length 小模型（768）要用bs 24
bs=16
# lr=3e-5
dropout=0.1
psl=20
epoch=30
task_type=sequence_classification
# DataTrainingArguments
# TrainingArguments
# ModelArguments
for lr in 9e-6 1e-5 3e-5 5e-5; do
    for seed in 10 20 30 40 50; do
        for DATASET_NAME in boolq; do
            for MODEL_NAME in bert-large-uncased; do
                for prompt_operation in attention; do
                    python3 ../../../run.py \
                      --task_name $TASK_NAME \
                      --dataset_name $DATASET_NAME \
                      --dataset_config_name None \
                      --max_seq_length 480 \
                      --overwrite_cache False \
                      --pad_to_max_length False \
                      --train_file None \
                      --validation_file None \
                      --test_file None \
                      --do_train \
                      --do_eval \
                      --per_device_train_batch_size $bs \
                      --per_device_eval_batch_size $bs \
                      --gradient_accumulation_steps 1 \
                      --num_train_epochs $epoch \
                      --pre_seq_len $psl \
                      --output_dir checkpoints/${task_type}_${TASK_NAME}_${DATASET_NAME}_${MODEL_NAME}_${prompt_operation}_${lr}_${seed}/ \
                      --overwrite_output_dir \
                      --learning_rate $lr \
                      --weight_decay 0.0005 \
                      --fp16 True \
                      --seed ${seed} \
                      --warmup_ratio 0.2 \
                      --save_strategy no \
                      --evaluation_strategy epoch \
                      --remove_unused_columns False \
                      --model_name_or_path ${MODEL_DIR}${MODEL_NAME} \
                      --use_fast_tokenizer True \
                      --model_revision main \
                      --prefix \
                      --task_type ${task_type} \
                      --prompt_type soft \
                      --pattern_id pattern_0 \
                      --template_id template_6 \
                      --verbalizer_id verbalizer_0 \
                      --prompt_operation ${prompt_operation}
                done
            done
        done
    done
done


