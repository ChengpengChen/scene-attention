CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 1 --dropout_ratio 0.5 --prefix 1bk_resnet50_mit67_bs16_drop0.5_force_resize_448a_1029 --set batch_size 16 aug.force_resize 448 image_size 448

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 5 --dropout_ratio 0.5 --prefix 5bk_resnet50_mit67_bs16_drop0.5_force_resize_448a_1029 --set batch_size 16 aug.force_resize 448 image_size 448

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 10 --dropout_ratio 0.5 --prefix 10bk_resnet50_mit67_bs16_drop0.5_force_resize_448a_1029 --set batch_size 16 aug.force_resize 448 image_size 448

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 1 --dropout_ratio 0.5 --prefix 1bk_resnet50_mit67_bs16_drop0.5_force_resize_336a_1029 --set batch_size 16 aug.force_resize 336 image_size 336

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 5 --dropout_ratio 0.5 --prefix 5bk_resnet50_mit67_bs16_drop0.5_force_resize_336a_1029 --set batch_size 16 aug.force_resize 336 image_size 336

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 10 --dropout_ratio 0.5 --prefix 10bk_resnet50_mit67_bs16_drop0.5_force_resize_336a_1029 --set batch_size 16 aug.force_resize 336 image_size 336

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 1 --dropout_ratio 0.5 --prefix 1bk_resnet50_mit67_bs16_drop0.5_force_resize_224a_1029 --set batch_size 16 aug.force_resize 224 image_size 224

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 5 --dropout_ratio 0.5 --prefix 5bk_resnet50_mit67_bs16_drop0.5_force_resize_224a_1029 --set batch_size 16 aug.force_resize 224 image_size 224

CUDA_VISIBLE_DEVICES=0 python finetune_v2.py --dataset mit67 --num_non_local_block 10 --dropout_ratio 0.5 --prefix 10bk_resnet50_mit67_bs16_drop0.5_force_resize_224a_1029 --set batch_size 16 aug.force_resize 224 image_size 224

