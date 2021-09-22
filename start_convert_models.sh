#!/bin/bash

python convert_vissl_to_torch.py \
    --model_url_or_file="checkpoints/model_checkpoints/swav_covid/model_phase950.torch" \
    --output_dir="vissl_models" \
    --output_name="swav_covid_e950"