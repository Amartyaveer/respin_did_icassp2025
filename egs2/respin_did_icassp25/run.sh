set -euo pipefail
y=s12345
z=bh
suffix="did"
gpu=0
x=${z}_${y}
./Run_indic_mono_did_baseline.sh --stage 2 \
	--stop_stage 13 \
	--gpu ${gpu} \
	--lang "${x}${suffix:+"_$suffix"}" \
	--tag "${x}${suffix:+"_$suffix"}" \
	--monolingual_asr_config "conf/tuning/train_asr_conformer_8enc_roberta_did_ml7-11.yaml" \
	--inference_config "conf/tuning/decode_lid.yaml" \
	--asr_tag "noaux_8enc_05ctc" \
	--asr_script asr_respin_lid_ce.sh \
	--aux_tag ""  \
	--data_folder data/data_${z} \
	--use_dial_file false \
	--train_dev "dev_${z}_nt${suffix:+"_$suffix"}" \
	--test_set "test_${z}_nt${suffix:+"_$suffix"}" \
	--asr_task "asr_roberta" \
	--tokenizer_path "ckpts_converted/bh_tokenizer.json" \
	--loss_scale_ce 7 \
	# --skip_train true \