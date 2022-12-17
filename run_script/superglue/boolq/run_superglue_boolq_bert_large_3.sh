# source activate
# conda activate py39

export TASK_NAME=superglue
# export DATASET_NAME=boolq
export CUDA_VISIBLE_DEVICES=3
# export MODEL_NAME=bert_base_uncased
export MODEL_DIR=/home/caizf/models/


# 大模型（1024）要用小学习率，1e-5，5e-6
# 小模型（768）用大学习率 2e-5
# 大模型参数锁定，prompt网络更新用大学习率，1e-2
# 16G显卡 128 max_seq_length 大模型（1024）要用bs
# 16G显卡 128 max_seq_length 小模型（768）要用bs 24
bs=8
# lr=3e-5
dropout=0.1
# psl=20
# psl=12
epoch=30
# DataTrainingArguments
# TrainingArguments
# ModelArguments
task_type=language_modeling
for lr in 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5 2e-5 3e-5 4e-5 5e-5; do
    for DATASET_NAME in boolq; do
        for MODEL_NAME in bert-large-cased; do
            for prompt_operation in attention; do
                python3 ../../../run.py \
                  --task_name $TASK_NAME \
                  --dataset_name $DATASET_NAME \
                  --dataset_config_name None \
                  --max_seq_length 512 \
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
                  --output_dir checkpoints/${task_type}_${TASK_NAME}_${DATASET_NAME}_${MODEL_NAME}_${prompt_operation}_${lr}_6_11/ \
                  --overwrite_output_dir \
                  --learning_rate $lr \
                  --weight_decay 0.0005 \
                  --seed 34 \
                  --save_strategy no \
                  --evaluation_strategy epoch \
                  --remove_unused_columns False \
                  --model_name_or_path ${MODEL_DIR}${MODEL_NAME} \
                  --use_fast_tokenizer True \
                  --model_revision main \
                  --prompt \
                  --task_type ${task_type} \
                  --prompt_type soft \
                  --pattern_id pattern_0 \
                  --template_id template_6 \
                  --verbalizer_id verbalizer_0 \
                  --prompt_operation ${prompt_operation} \
                  --prefix_projection False \
                  --hidden_dropout_prob $dropout \
                  --num_attention_layers 6 \
                  --num_attention_heads 8 \
                  --whether_PositionalEncoding True \
                  --whether_PositionalWiseFeedForward True
            done
        done
    done
done


# python3 run.py \
#   --model_name_or_path bert-base-cased \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$TASK_NAME_$DATASET_NAME_bert/ \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prefix
