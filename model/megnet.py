import torch
from torch_geometric.nn import GATConv
from torch import optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import lightning.pytorch as pl
from sklearn.metrics import r2_score
from deepchem.models.torch_models.layers import GraphNetwork as GN
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU

# __________________
# Data Preprocessing
def preprocess(data, args):
    if args.normalize:
        # =====================
        # Scale node attributes
        # =====================
        x_scalers = data.x_scalers
        for i in range(data.x.shape[1]):
            data.x[:, i] = torch.tensor(torch.log1p(data.x[:, i].reshape(-1, 1)).reshape(-1))
        data.x = data.x.float()

        # ==========================
        # Scale conditional features
        # ==========================
        conds_scalers = data.cond_feat_scalers
        for i in range(data.cond_feat.shape[1]):
            cond_scaler = conds_scalers[0]
            if cond_scaler is None:
                data.cond_feat[:, i] = torch.zeros_like(data.cond_feat[:, i])
            else:
                if data.batch != None:
                    data.cond_feat[:, i] = torch.tensor(conds_scalers[0][i].transform(data.cond_feat[:, i].reshape(-1, 1).cpu().numpy()).reshape(-1))
                else:
                    data.cond_feat[:, i] = torch.tensor(conds_scalers[i].transform(data.cond_feat[:, i].reshape(-1, 1).cpu().numpy()).reshape(-1))
        data.cond_feat = data.cond_feat.float()

        # ==============
        # Scale y values
        # ==============
        data.y = torch.tensor(torch.log1p(data.y)).float()

    return data

class Model(pl.LightningModule):
    def __init__(self, args, sample_data):
        super(Model, self).__init__()
        self.args = args
        sample_data = preprocess(sample_data, args)
        self.encoder = GN(n_node_features=sample_data.x.shape[1],
                   n_edge_features=sample_data.edge_attr.shape[1],
                   n_global_features=sample_data.cond_feat.shape[1],
                   is_undirected=True,
                   residual_connection=True)
        
        self.decoder = Sequential(Linear(sample_data.x.shape[1] , args.hidden_dim),
                            ReLU(),
                            Linear(args.hidden_dim, 1),
                            )
        
        self.training_step_outputs = []   
        self.training_step_targets = []   
        self.val_step_outputs = []        
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def loss(self, pred, inputs):
        mse = nn.functional.mse_loss(pred, inputs.y)
        rmse = torch.sqrt(mse)
        mae = nn.functional.l1_loss(pred, inputs.y)
        r2 = r2_score(inputs.y.cpu().detach().numpy(), pred.cpu().detach().numpy())

        return rmse, mae, r2
    
    def get_epoch_results(self, outputs, targets):
        outputs = np.array(outputs)
        targets = np.array(targets)

        rmse = np.sqrt(mean_squared_error(targets, outputs))
        mae = np.mean(np.abs(outputs - targets))
        r2 = r2_score(targets, outputs)

        return rmse, mae, r2
    
    def forward(self, data):
        data = preprocess(data, self.args)
        h, edge_h, global_h = self.encoder(node_features=data.x, edge_index=data.edge_index, edge_features=data.edge_attr, global_features=data.cond_feat, batch=data.batch)

        pred = self.decoder(h)

        return pred
        
    def training_step(self, batch, batch_idx):
        pred = self(batch)
        rmse, mae, r2 = self.loss(pred, batch)

        self.log("Train RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Train R2", r2, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_step=False, on_epoch=True)

        return rmse

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        rmse, mae, r2 = self.loss(pred, batch)

        y_pred = pred.cpu().numpy()
        y_true = batch.y.cpu().numpy()
        
        self.val_step_outputs.extend(list(y_pred.flatten()))
        self.val_step_targets.extend(list(y_true.flatten()))

        return rmse
    
    def on_validation_epoch_end(self):
        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets
        rmse, mae, r2 = self.get_epoch_results(val_all_outputs, val_all_targets)
        self.log("Valid RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Valid MAE", mae, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Valid R2", r2, prog_bar=True, batch_size=self.args.batch_size)

        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        rmse, mae, r2 = self.loss(pred, batch)

        y_pred = pred.cpu().numpy()
        y_true = batch.y.cpu().numpy()
        
        self.test_step_outputs.extend(list(y_pred.flatten()))
        self.test_step_targets.extend(list(y_true.flatten()))

        return rmse

    def on_test_epoch_end(self):
        test_all_outputs = self.test_step_outputs
        test_all_targets = self.test_step_targets
        rmse, mae, r2 = self.get_epoch_results(test_all_outputs, test_all_targets)
        self.log("Test RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Test MAE", mae, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Test R2", r2, prog_bar=True, batch_size=self.args.batch_size)

        self.test_step_outputs.clear()
        self.test_step_targets.clear()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]