#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=4
stop_stage=13
suffix="indic_char"
nbpe=500
lang=bn_did
gpu=1
ngpu=0
tag="bn_did"

asr_script=asr_respin_lid.sh
monolingual_asr_config=conf/tuning/train_asr_conformer_scctc_4enc.yaml
multilingual_asr_config=conf/tuning/train_asr_conformer_hier_lid_tok_indic.yaml
inference_config=conf/tuning/decode_lid.yaml
tokenizer_path=data_bn/tokenizer_${nbpe}.model
loss_scale_ce=1.0
debug=false

asr_tag="auxctc_4enc_did"
aux_tag="lid_utt"
asr_task="asr"
infer_tag="decode_lid_asr_model_valid.acc.ave"
data_folder=data_bn

infer_model="valid.acc.ave.pth"
train_nj=1
use_dial_file=false


train_dev=dev_bn
test_set=test_bn
skip_train=false
gpu_infer=false

. utils/parse_options.sh

export CUDA_VISIBLE_DEVICES="$gpu"

train_set=train_${tag}

nlsyms_txt=${data_folder}/nlsyms.txt
lm_config=conf/train_lm.yaml


./${asr_script} --stage $stage --stop_stage $stop_stage \
    --tag "${tag}" \
    --data_folder "${data_folder}" \
    --expdir "exp/exp_${tag}${suffix:+_$suffix}" \
    --dumpdir "dump/dump_${tag}" \
    --lang "${tag}" \
    --auxiliary_data_tags "${aux_tag}" \
    --local_data_opts "--stage 0 --lang ${lang} --nlsyms_txt ${nlsyms_txt}" \
    --post_process_local_data_opts "--stage 2 --lang ${lang} --nlsyms_txt ${nlsyms_txt}" \
    --audio_format "wav" \
    --use_lm false \
    --feats_normalize utt_mvn \
    --lm_config "${lm_config}" \
    --token_type char \
    --nbpe $nbpe \
    --bpe_nlsyms "${nlsyms_txt}" \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${monolingual_asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "${data_folder}/${train_set}/text" \
    --lm_train_text "${data_folder}/${train_set}/text" \
    --local_score_opts "--score_lang_id true" "$@" \
    --nj $train_nj \
    --ngpu $ngpu \
    --inference_asr_model ${infer_model} \
    --inference_nj 16 \
    --asr_tag "$asr_tag" \
    --inference_tag "$infer_tag" \
    --use_dial_file ${use_dial_file} \
    --skip_train ${skip_train} \
    --asr_task "${asr_task}" \
    --tokenizer_path "${tokenizer_path}" \
    --loss_scale_ce ${loss_scale_ce} \
    --gpu_inference ${gpu_infer} \
    --debug ${debug} \
    || exit 1
