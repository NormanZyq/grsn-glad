# aids
python src/train.py -m seed=1,12,123,1234,12345 experiment=aids_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=aids_grsnc2_ablation_random_str trainer=gpu logger=wandb

# proteins
python src/train.py -m seed=1,12,123,1234,12345 experiment=proteins_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=proteins_grsnc2_ablation_random_str trainer=gpu logger=wandb

# hse
python src/train.py -m seed=1,12,123,1234,12345 experiment=hse_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=hse_grsnc2_ablation_random_str trainer=gpu logger=wandb

# p53
python src/train.py -m seed=1,12,123,1234,12345 experiment=p53_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=p53_grsnc2_ablation_random_str trainer=gpu logger=wandb

# dhfr
python src/train.py -m seed=1,12,123,1234,12345 experiment=dhfr_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=dhfr_grsnc2_ablation_random_str trainer=gpu logger=wandb

# mmp
python src/train.py -m seed=1,12,123,1234,12345 experiment=mmp_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=mmp_grsnc2_ablation_random_str trainer=gpu logger=wandb

# reddit
#python src/train.py -m seed=1,12,123,1234,12345 experiment=reddit_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=reddit_grsnc2_ablation_random_str trainer=gpu logger=wandb

# imdb
#python src/train.py -m seed=1,12,123,1234,12345 experiment=imdb_grsnc2_ablation_no_str trainer=gpu logger=wandb
python src/train.py -m seed=1,12,123,1234,12345 experiment=imdb_grsnc2_ablation_random_str trainer=gpu logger=wandb