#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


MODEL_NAME_OR_PATH="/data/Qwen2.5-0.5B-Instruct" # model path

TRAIN_DATASETS="/data/align_anything_t2t" # rm dataset path
TRAIN_TEMPLATE="HOMEWORK" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_ROOT_DIR="../hsy_0528_outputs_dpo"

if [ -z "$OUTPUT_ROOT_DIR" ]; then
    echo "OUTPUT_ROOT_DIR is not set"
    OUTPUT_ROOT_DIR="../outputs"
fi

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/qwen_2_5_dpo" # output dir

# For wandb online logging
export WANDB_API_KEY="bcaa59392385ab52966422a94b30994485350d7a"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.dpo \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --epochs 1 