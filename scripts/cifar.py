import os
# Ours
os.system("python test_agr.py --dataset cifar10 --gpu 6 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation median --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")

os.system("python test_agr.py --dataset cifar10 --gpu 4 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")

os.system("python test_agr.py --dataset cifar10 --gpu 5 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")

