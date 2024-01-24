python bdfg_distributed_stable.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20

python ef22_scaled_randK.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20

python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20  --method EF21 --compressor SRandK