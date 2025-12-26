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
import os
from scipy.io import loadmat

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
        self._beta_cache = {}  # case_number -> torch.Tensor [M, M]
        sample_data = preprocess(sample_data, args)
        self.encoder = GATConv(in_channels=(sample_data.x.shape[1]+sample_data.cond_feat.shape[1]+1), out_channels=args.hidden_dim, heads=1)
        self.decoder = Sequential(Linear(args.hidden_dim, args.hidden_dim),
                            ReLU(),
                            Linear(args.hidden_dim, 1),
                            )
        
        self.training_step_outputs = []   
        self.training_step_targets = []   
        self.val_step_outputs = []        
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def _get_case_numbers(self, batch):
        if not hasattr(batch, "case_number"):
            raise RuntimeError("Batch has no 'case_number'. Required to load beta matrices.")
        cn = batch.case_number
        if torch.is_tensor(cn):
            return [int(x) for x in cn.view(-1).tolist()]
        if isinstance(cn, (list, tuple)):
            return [int(x) for x in cn]
        return [int(cn)] * batch.num_graphs
    
    def _load_beta_from_dir(self, case_number: int, M_expected: int, device, dtype=torch.float32):
        if case_number in self._beta_cache:
            beta = self._beta_cache[case_number]
            if beta.shape != (M_expected, M_expected):
                raise RuntimeError(
                    f"[beta cache shape mismatch] case {case_number}: "
                    f"cached {tuple(beta.shape)} vs expected {(M_expected, M_expected)}"
                )
            return beta.to(device=device, dtype=dtype)

        fname = f"beta_term_case{int(case_number)}.mat"
        path = os.path.join(self.args.beta_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"beta file not found: {path}")

        mat = loadmat(path)
        for key in ("betamn_particle", "beta", "betamn", "beta_matrix"):
            if key in mat:
                beta_np = mat[key]
                break
        else:
            raise KeyError(
                f"No expected beta key in {path}. "
                "Tried: betamn_particle, beta, betamn, beta_matrix"
            )

        if beta_np.ndim != 2 or beta_np.shape[0] != beta_np.shape[1]:
            raise ValueError(f"beta in {path} must be square 2D, got {beta_np.shape}")
        if beta_np.shape != (M_expected, M_expected):
            raise ValueError(
                f"beta shape mismatch in {path}: got {beta_np.shape}, expected {(M_expected, M_expected)}. "
                "Ensure graph pivot count M equals the beta size."
            )

        beta = torch.tensor(beta_np, dtype=dtype, device=device)
        self._beta_cache[case_number] = beta
        return beta
    
    def _make_pivot_v_from_r0(self, r0_per_graph: torch.Tensor, M: int, s: float = 1.12):
        """
        r0_per_graph: [B]
        return: v [B, M]  
        """
        B = r0_per_graph.shape[0]
        i = torch.arange(M, device=r0_per_graph.device, dtype=r0_per_graph.dtype)  # 0..M-1
        r = r0_per_graph.view(B, 1) * (s ** i.view(1, M))        # [B, M], r_i = r0 * s^i
        v = (4.0/3.0) * torch.pi * (r ** 3)                      # [B, M], nm^3
        return v
    
    def aggregation_birth_death_fixed_pivot(self, N, beta_matrix, v):
        """
        N:   [B, M]
        beta:[B, M, M]
        v:   [B, M]
        Returns: B_term, D_term each [B, M]
        """
        Bsz, M = N.shape

        # Death: D_i = N_i * sum_k beta_{i,k} N_k
        sum_term = torch.einsum('bij,bj->bi', beta_matrix, N)
        D = N * sum_term

        # Birth: fixed-pivot with eta + (1 - 0.5*delta_jk)
        B_term = torch.zeros_like(N)
        for i in range(M):
            v_i = v[:, i]
            v_im1 = v[:, i - 1] if i - 1 >= 0 else None
            v_ip1 = v[:, i + 1] if i + 1 < M else None

            for j in range(M):
                for k in range(j, M):  # j <= k (dedup)
                    vjk = v[:, j] + v[:, k]
                    eta = torch.zeros(Bsz, device=N.device, dtype=N.dtype)

                    if i + 1 < M:
                        maskA = (v_i <= vjk) & (vjk <= v_ip1)
                        denomA = (v_ip1 - v_i).clamp_min(1e-12)
                        eta[maskA] = (v_ip1[maskA] - vjk[maskA]) / denomA[maskA]

                    if i - 1 >= 0:
                        maskB = (v_im1 <= vjk) & (vjk <= v_i)
                        denomB = (v_i - v_im1).clamp_min(1e-12)
                        eta[maskB] = (vjk[maskB] - v_im1[maskB]) / denomB[maskB]

                    if torch.all(eta == 0):
                        continue

                    w = 0.5 if j == k else 1.0
                    B_term[:, i] += w * eta * beta_matrix[:, j, k] * N[:, j] * N[:, k]

        return B_term, D
    
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
        cond_feat_expanded = data.cond_feat[data.batch]  # (total_num_nodes, d)
        target_timestep_expanded = data.target_timestep[data.batch].unsqueeze(1)  # (total_num_nodes, d)
        node_feat = torch.cat((data.x, cond_feat_expanded, target_timestep_expanded), dim=1)

        data.x = node_feat

        # Generate isolated edge index for each graph
        data.edge_index = torch.empty((2, 0), dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
        
        # No edges
        pred = self.decoder(self.encoder(x=data.x, edge_index=data.edge_index))

        return pred
        
    # -------------------------
    # PINN residual
    # -------------------------
    def compute_pbe_residual_loss(self, pred_for_N, dlogN_dt_per_node, batch):
        B = batch.num_graphs
        M = pred_for_N.size(0) // B

        # log1p(N) -> N
        N = torch.expm1(pred_for_N.view(B,M))  # shape: [N_total, 1]

        # beta per graph from directory
        case_nums = self._get_case_numbers(batch)
        beta_list = [
            self._load_beta_from_dir(cn, M_expected=M, device=N.device, dtype=N.dtype)
            for cn in case_nums
        ]
        beta = torch.stack(beta_list, dim=0)  # [B, M, M]

        r0 = (torch.expm1(batch.x[batch.ptr[:-1]][:,0])/2).clone().detach()
        if r0.dim() != 1 or r0.numel() != batch.num_graphs:
            raise ValueError(f"batch.r0 must be shape [B]; got {tuple(r0.shape)}")
        v = self._make_pivot_v_from_r0(r0_per_graph=r0, M=M, s=1.12)  # [B, M]

        B_term, D_term = self.aggregation_birth_death_fixed_pivot(N=N, beta_matrix=beta, v=v)  # each [B, M]
        dN_dt_per_node = dlogN_dt_per_node * (N + 1)  # [B, M]
        residual_per_node = dN_dt_per_node - (B_term - D_term)  # [B, M]

        # ---- mask out t=0 graphs from physics loss (since IC is enforced as a hard constraint) ----
        # data.target_timestep: [B]; consider |t|<=tol as t=0
        t_graph = batch.target_timestep.float().view(-1)  # [B]
        t_graph = t_graph - 10 
        non_ic_mask = (t_graph.abs() > 1e-12).view(B, 1)  # True for t != 0
        residual_per_node = residual_per_node * non_ic_mask  # zero out IC rows

        return torch.sqrt(torch.mean(residual_per_node ** 2) + 1e-10)
    
    def training_step(self, batch, batch_idx):

        if batch.target_timestep.requires_grad is False: 
            batch.target_timestep = batch.target_timestep.float().requires_grad_(True) 

        # 2. Forward 패스 실행  
        pred = self(batch)

        # 3. 손실 계산
        rmse, mae, r2 = self.loss(pred, batch)

        (grad_wrt_node_feat,) = torch.autograd.grad(
            outputs=pred,
            inputs=batch.x,  
            grad_outputs=torch.ones_like(pred), 
            retain_graph=True, 
            create_graph=True
        )

        dlogN_dt_per_node = grad_wrt_node_feat[:, -1].reshape(batch.num_graphs, -1) # shape: [B, N_total]

        rmse_phys = self.compute_pbe_residual_loss(pred, dlogN_dt_per_node, batch)
        total_loss = rmse + self.args.lambda_phys * rmse_phys

        self.log("Train RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Train R2", r2, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Train Phys RMSE", rmse_phys, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Train Total Loss", total_loss, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_step=False, on_epoch=True)

        return total_loss

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