# CUDA_VISIBLE_DEVICES=3 python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_sun397_bs16_drop0.5_1015 --set batch_size 16

CUDA_VISIBLE_DEVICES=1 python finetune_v2.py --dataset mit67 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_mit67_bs16_drop0.5_resize_shorter_384_1027 --set batch_size 16 aug.resize_shorter 384 image_size 336

CUDA_VISIBLE_DEVICES=1 python finetune_v2.py --dataset mit67 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_mit67_bs16_drop0.5_resize_shorter_512_1027 --set batch_size 16 aug.resize_shorter 512 image_size 448

