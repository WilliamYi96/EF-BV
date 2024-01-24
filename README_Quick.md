python generate_data.py --dataset mushrooms --num_workers 20 --loss_func log_reg
python generate_data.py --dataset mushrooms --num_workers 50 --loss_func log_reg

python generate_data.py --dataset w8a --num_workers 20 --loss_func log_reg
python generate_data.py --dataset w8a --num_workers 50 --loss_func log_reg

python generate_data.py --dataset a9a --num_workers 20 --loss_func log_reg
python generate_data.py --dataset a9a --num_workers 50 --loss_func log_reg

python generate_data.py --dataset phishing --num_workers 20 --loss_func log_reg
python generate_data.py --dataset phishing --num_workers 50 --loss_func log_reg



python generate_data_overlap.py --dataset mushrooms --num_workers 20 --times 1          
python generate_data_overlap.py --dataset mushrooms --num_workers 20 --times 2

python generate_data_overlap.py --dataset w8a --num_workers 50 --times 1            
python generate_data_overlap.py --dataset w8a --num_workers 50 --times 2

python generate_data_overlap.py --dataset a9a --num_workers 20 --times 1          
python generate_data_overlap.py --dataset a9a --num_workers 20 --times 2

python generate_data_overlap.py --dataset phishing --num_workers 50 --times 1            
python generate_data_overlap.py --dataset phishing --num_workers 50 --times 2     

