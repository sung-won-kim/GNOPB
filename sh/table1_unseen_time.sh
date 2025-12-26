cd ../

for model in gnopb megnet mlp
    do
    python run_table1_unseen_time.py --model $model --summary Table1_unseen_time --epochs 100 --devices '0' --lr 0.001 --seed 0 --num_seed 1 --wb_name ${model}
    done