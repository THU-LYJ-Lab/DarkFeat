# python test_rgb.py --task=pretrained \
#                 --data_path="./data/" \
#                 --gamma \
#                 --camera="Canon_EOS_5D" \
#                 --out_path="./exps/" \
#                 --ckpt="./pretrained/canon.pth" \
#                 # --split_to_patch

python test_raw.py --task=pretrained \
                --data_path="./data/" \
                --gamma \
                --camera="Canon_EOS_5D" \
                --out_path="./exps/" \
                --ckpt="./pretrained/canon.pth" \
                --split_to_patch
