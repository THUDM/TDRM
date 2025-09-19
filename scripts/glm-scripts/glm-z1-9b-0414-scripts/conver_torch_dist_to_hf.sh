PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --model-name glm4 \
    --input-dir /root/slime/glm-z1-ckpt/glm_scalarprm/iter_0000099 \
    --output-dir /root/slime/glm-z1-ckpt/hf_glm_scalarprm/iter_0000099
