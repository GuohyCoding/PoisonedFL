import os
# Ours
os.system("python test_agr.py --dataset mnist --gpu 7 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation median --byz_type poisonedfl --sf 8 --local_epoch 1 ")

os.system("python test_agr.py --dataset mnist --gpu 8 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 ")

os.system("python test_agr.py --dataset mnist --gpu 3 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 ")
