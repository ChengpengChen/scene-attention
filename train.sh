python finetune_v2.py --num_non_local_block 0 --prefix baseline_resnet50_mit67_bs16_0906

python finetune_v2.py --num_non_local_block 0 --prefix baseline_resnet50_mit67_bs32_0906 --set batch_size 32

python finetune_v2.py --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_mit67_bs16_drop0.5_0906

python finetune_v2.py --num_non_local_block 0 --dropout_ratio 0.5 --prefix baseline_resnet50_mit67_bs32_drop0.5_0906 --set batch_size 32

python finetune_v2.py --num_non_local_block 1 --prefix resnet50_mit67_bs32_1bk_0906 --set batch_size 32

python finetune_v2.py --num_non_local_block 1 --dropout_ratio 0.5 --prefix resnet50_mit67_bs32_1bk_drop0.5_0906 --set batch_size 32


python finetune_v2.py --num_non_local_block 5 --prefix resnet50_mit67_bs16_5bk_0906

python finetune_v2.py --num_non_local_block 5 --dropout_ratio 0.5 --prefix resnet50_mit67_bs16_5bk_drop0.5_0906

python finetune_v2.py --num_non_local_block 5 --prefix resnet50_mit67_bs32_5bk_0906 --set batch_size 32

python finetune_v2.py --num_non_local_block 5 --dropout_ratio 0.5 --prefix resnet50_mit67_bs32_5bk_drop0.5_0906 --set batch_size 32



python finetune_v2.py --num_non_local_block 10  --prefix resnet50_mit67_bs16_10bk_0906

python finetune_v2.py --num_non_local_block 10  --dropout_ratio 0.5 --prefix resnet50_mit67_bs16_10bk_drop0.5_0906

python finetune_v2.py --num_non_local_block 10 --prefix resnet50_mit67_bs32_10bk_0906 --set batch_size 32

python finetune_v2.py --num_non_local_block 10 --dropout_ratio 0.5 --prefix resnet50_mit67_bs32_10bk_drop0.5_0906 --set batch_size 32

