python3 -u main.py --train_pg --episode=5000 -lr=0.003 --model_name=lr0003 | tee lr0003.txt

python3 -u main.py --train_pg --episode=5000 -lr=0.003 --model_name=lr0003_v1 | tee lr0003_v1.txt

python3 -u main.py --train_pg --episode=5000 -lr=0.003 --model_name=lr0003_v2 | tee lr0003_v2.txt

python3 -u main.py --train_pg --episode=5000 -lr=0.001 --model_name=lr0001 | tee lr0001.txt

python3 -u main.py --train_pg --episode=5000 -lr=0.001 --model_name=lr0001_v1 | tee lr0001_v1.txt
