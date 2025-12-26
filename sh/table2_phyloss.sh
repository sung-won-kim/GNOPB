cd ../

# case num: IS, zeta, r_init, N0
# case 1: 77.7 / -14.3 / 131.5 / 7.64x10^13
# case 112: 15.9 / -26.8 / 739.1 / 3.06x10^13
# case 207: 20.1 / -32.7 / 173.6 / 2.54x10^14
# case 416: 59.5 / -29.5 / 646.1 / 3.06x10^14
# case 600: 31.1 / -12.0 / 406.9 / 1.58x10^16

hidden_dim=128
for model in gnopb mlp mlp_pinn
    do
        for case_id in 1 112 207 416 600
        do
            for lambda_phys in 1.0
            do 
            python run_table2_phyloss.py --model $model --summary Table2 --epochs 100 --devices '0' --batch_size 1 --lr 0.001 --seed 0 --num_seed 1 --wb_name ${model}_case${case_id} --case_id $case_id
            done
        done
    done
