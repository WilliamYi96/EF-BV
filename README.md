# EF-BV
This is the official code repository for NeurIPS 2022 paper: [EF-BV: A Unified Theory of Error Feedback and Variance Reduction Mechanisms for Biased and Unbiased Compression in Distributed Optimization](https://arxiv.org/abs/2205.04180)

## Data Preparation
We first split the original data:

```angular2html
python generate_data.py --dataset mushrooms|w8a|a9a|phishing --num_workers 20|50 --loss_func least_square|log-reg
```
 
Then considering overlapping distribution:

```angular2html
python generate_data_overlap.py --num_workers 20|50 --times 1|2|3|5|20|50
```

## Reproduce Appendix. A3
### mushrooms
```
python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 4 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 64 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 512 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1024 --num_workers 20
```


## Reproduce 5.2 - Logistic Regression

### mushrooms
```
python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 2 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 4 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 8 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 16 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 32 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 64 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 128 --num_workers 20
```


### w8a
```
python bdfg_distributed_stable.py --k 1 --dataset w8a --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset w8a --max_it 10000 --tol 1e-7 --factor 8 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset w8a --max_it 10000 --tol 1e-7 --factor 16 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset w8a --max_it 10000 --tol 1e-7 --factor 32 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset w8a --max_it 10000 --tol 1e-7 --factor 64 --num_workers 20

python bdfg_distributed_stable.py --k 1 --dataset w8a --max_it 10000 --tol 1e-7 --factor 128 --num_workers 20
```


### Rand-K
```
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 2 --num_workers 20
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 4 --num_workers 20
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 8 --num_workers 20
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 16 --num_workers 20
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 32 --num_workers 20
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 64 --num_workers 20
python bdfg_distributed_stable_rand_k.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 128 --num_workers 20

python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20
python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 2 --num_workers 20
python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 4 --num_workers 20
python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 8 --num_workers 20
python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 16 --num_workers 20
python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 32 --num_workers 20
python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 64 --num_workers 20
python bdfg_distributed_stable_rand_k_dep.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 128 --num_workers 20
```



## Reproduce CIFAR10 Image Classification
```
python EF21_100K.py --factor 8 --max_it 4545 --k 1320000 --batch_size 128 --model vgg11 --dataset CIFAR10
```



## Citation
```
@article{efbv2022,
  title={EF-BV: A unified theory of error feedback and variance reduction mechanisms for biased and unbiased compression in distributed optimization},
  author={Condat, Laurent and Yi, Kai and Richt{\'a}rik, Peter},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={17501--17514},
  year={2022}
}
```