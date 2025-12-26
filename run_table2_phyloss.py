import os
import argparse
import torch
import numpy as np
import wandb
import time
import random
import importlib
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader as PyGDataLoader
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main(args):

    if args.wb_name == None :
        args.wb_name = f'{args.model}_case{args.case_id}'

    for s_id, seed in enumerate(range(args.seed, args.seed + args.num_seed)):

        torch.manual_seed(seed)  #Torch
        random.seed(seed)        #Python
        np.random.seed(seed)     #NumPy
        
        module = importlib.import_module(f"model.{args.model}")
        Model = getattr(module, "Model")

        # Load data
        G_list = torch.load('./data/graph_data.pt')

        case_numbers = [data['case_number'] for data in G_list]
        case_types = torch.tensor(case_numbers).unique()

        # Split dataset
        case_types = case_types[torch.randperm(len(case_types))]

        selected_case = [args.case_id]
        
        train_target_step = [10, 20, 30]
        val_target_step = [40, 50]
        test_target_step = [60, 70, 80, 90, 100, 110, 120]

        train_g_list, val_g_list, test_g_list = [], [], []
        for data in G_list:
            case_number = data['case_number']
            if case_number in selected_case and data.target_timestep in train_target_step:
                train_g_list.append(data)
            elif case_number in selected_case and data.target_timestep in val_target_step:
                val_g_list.append(data)
            elif case_number in selected_case and data.target_timestep in test_target_step:
                test_g_list.append(data)
        
        class DataModule(L.LightningDataModule):
            def train_dataloader(self):
                return PyGDataLoader(train_g_list, batch_size=args.batch_size, num_workers=1, shuffle=True)
            def val_dataloader(self):
                return PyGDataLoader(val_g_list, batch_size=args.batch_size, num_workers=1, shuffle=False)
            def test_dataloader(self):
                return PyGDataLoader(test_g_list, batch_size=args.batch_size, num_workers=1, shuffle=False)
            def predict_dataloader(self):
                return PyGDataLoader(test_g_list, batch_size=args.batch_size, num_workers=1, shuffle=False)

        wandb_logger = WandbLogger(project=f'{args.summary}', name=f'{args.wb_name}')
        dirpath = f'./best_models/{args.summary}/{args.wb_name}/s{seed}_{time.strftime("%m%d-%H%M")}/'
        checkpoint_callback = ModelCheckpoint(
            monitor='Valid RMSE',
            mode="min",
            dirpath=dirpath,
            filename='best',
            save_top_k=1,
            save_last=True)
        
        if not os.path.exists(dirpath): 
            os.makedirs(dirpath) 

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=args.devices,
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=5), LearningRateMonitor(logging_interval='epoch')],
            log_every_n_steps=1, 
            logger=wandb_logger,
            check_val_every_n_epoch=args.val_interval,
        )

        model = Model(args, G_list[0].clone())

        datamodule = DataModule()

        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(datamodule=datamodule, ckpt_path='best')

        wandb.finish()

if __name__ == '__main__':

    timestr = time.strftime("%m$d")

    def list_of_ints(arg):
        if arg == 'cpu':
            return arg
        else:
            return list(map(int, arg.split(',')))
        
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--model", type=str, default='mlp_timeadded_pinn')
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--epochs", type=int, default=300)
        parser.add_argument("--devices", type=list_of_ints, default='0')
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--val_interval", type=int, default=5)
        parser.add_argument("--summary", type=str, default=f'[table2]_{timestr}')
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--num_seed", type=int, default=1)
        parser.add_argument("--wb_name", type=str, default=None)  # 50
        parser.add_argument("--normalize", type=str, default=1) 
        parser.add_argument("--return_weight", type=str, default=0) 
        # PINN args
        parser.add_argument("--lambda_data", type=float, default=1.0)
        parser.add_argument("--lambda_phys", type=float, default=0.1)
        parser.add_argument("--beta_dir", type=str, default='./data/beta_matrices')
        parser.add_argument("--case_id", type=int, default=1)



        return parser.parse_known_args()

    args, unknown = parse_args()

    main(args)
