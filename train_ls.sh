# CUDA_VISIBLE_DEVICES=2 python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_sun397_bs16_drop0.5_1025 --set batch_size 16

# CUDA_VISIBLE_DEVICES=2 python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_sun397_bs16_drop0.5_resize_shorter_384_1025 --set batch_size 16 aug.force_resize 0 aug.resize_shorter 384 image_size 336

# CUDA_VISIBLE_DEVICES=2 python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_sun397_bs16_drop0.5_force_resize_384_1025 --set batch_size 16 aug.force_resize 384 image_size 336

# CUDA_VISIBLE_DEVICES=2 python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_sun397_bs16_drop0.5_resize_shorter_1025 --set batch_size 16 aug.force_resize 0 aug.resize_shorter 256

CUDA_VISIBLE_DEVICES=1 python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_sun397_bs16_drop0.5_force_resize_336a_1025 --set batch_size 16 aug.force_resize 336 image_size 336

CUDA_VISIBLE_DEVICES=1 python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_sun397_bs16_drop0.5_force_resize_224a_1025 --set batch_size 16 aug.force_resize 224 image_size 224

