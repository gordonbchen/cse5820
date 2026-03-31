python3 -u main.py --train_pg --episode=5000 -lr=0.001 --model_name=lr0001 | tee runs/lr0001/log.txt

python3 -u main.py --train_pg --episode=5000 -lr=0.003 --model_name=lr0003_2 | tee runs/lr0003_2/log.txt
