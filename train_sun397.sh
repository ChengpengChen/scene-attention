python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5  --prefix baseline_resnet50_sun397_bs16_drop0.5_0903 --set batch_size 16

python finetune_v2.py --dataset sun397 --num_non_local_block 0 --dropout_ratio 0.5  --prefix baseline_resnet50_sun397_bs32_drop0.5_0903 --set batch_size 32

python finetune_v2.py --dataset sun397 --num_non_local_block 1 --prefix resnet50_sun397_bs32_1bk_0903 --set batch_size 32

python finetune_v2.py --dataset sun397 --num_non_local_block 1 --dropout_ratio 0.5 --prefix resnet50_sun397_bs32_1bk_drop0.5_0903 --set batch_size 32


python finetune_v2.py --dataset sun397 --num_non_local_block 5 --prefix resnet50_sun397_bs16_5bk_0903

python finetune_v2.py --dataset sun397 --num_non_local_block 5 --dropout_ratio 0.5 --prefix resnet50_sun397_bs16_5bk_drop0.5_0903

python finetune_v2.py --dataset sun397 --num_non_local_block 5 --prefix resnet50_sun397_bs32_5bk_0903 --set batch_size 32

python finetune_v2.py --dataset sun397 --num_non_local_block 5 --dropout_ratio 0.5 --prefix resnet50_sun397_bs32_5bk_drop0.5_0903 --set batch_size 32



python finetune_v2.py --dataset sun397 --num_non_local_block 10  --prefix resnet50_sun397_bs16_10bk_0903

python finetune_v2.py --dataset sun397 --num_non_local_block 10  --dropout_ratio 0.5 --prefix resnet50_sun397_bs16_10bk_drop0.5_0903

python finetune_v2.py --dataset sun397 --num_non_local_block 10 --prefix resnet50_sun397_bs32_10bk_0903 --set batch_size 32

python finetune_v2.py --dataset sun397 --num_non_local_block 10 --dropout_ratio 0.5 --prefix resnet50_sun397_bs32_10bk_drop0.5_0903 --set batch_size 32

