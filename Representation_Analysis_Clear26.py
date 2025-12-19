# Auto-generated from Representation_Analysis_Clear26.ipynb
# Notebook cells have been linearized into a plain Python script.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.autograd import Function
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
torch.manual_seed(0)
import os
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from scipy.special import digamma
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from numpy.random import default_rng
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, log_loss
from sklearn.decomposition import PCA

# Optional embed backends
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

# Optional SciPy (fast MI)
_HAS_SCIPY = True
try:
    from scipy.spatial import cKDTree
    from scipy.special import digamma
except Exception:
    _HAS_SCIPY = False


# In[2]:


def set_global_seed(seed: int = 0):
    # 1) Python hashing & libraries
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) Control BLAS/OpenMP threads for determinism (optional but helps)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # 3) NumPy
    import numpy as _np
    _np.random.seed(seed)

    # 4) (If you use PyTorch somewhere)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # If you want strict determinism (may error for unsupported ops):
        # torch.use_deterministic_algorithms(True)
        # For CUDA BLAS determinism (PyTorch >= 1.8):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


def add_dummy_features_shuffle(data, no_features):
    np.random.seed(2)
    data_dummy=pd.DataFrame()
    dummies=no_features

    if(dummies==0):
        return data

    elif (dummies==1):
        #data_aux = pd.DataFrame({f"d{0}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
        #arr=data.iloc[:,5].values
        #data_aux = pd.DataFrame({f"d{0}"  : data.iloc[:,5].sample(frac=1).reset_index(drop=True)})
        arr=data.iloc[:,5].values
        np.random.shuffle(arr)
        data_aux = pd.DataFrame({f"d{0}"  :arr.tolist() })
        #data_aux = pd.DataFrame({f"d{0}"  :np.random.shuffle(arr)},index=[0])
        data_dummy = pd.concat([data_dummy, data_aux],axis=1)
    else:
        for i in range(5,5+dummies):
            #np.random.seed(i)
            #data_aux = pd.DataFrame({f"d{i}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
            #data_aux = pd.DataFrame({f"d{i}"  : data.iloc[:,5].sample(frac=1).reset_index(drop=True)})
            arr=data.iloc[:,5].values #22
            #arr=np.random.normal(0,1, size=data.shape[0])
            np.random.shuffle(arr)
            data_aux = pd.DataFrame({f"d{i}"  :arr.tolist() })
            data_dummy = pd.concat([data_dummy, data_aux],axis=1)

    new_data=pd.concat([data,data_dummy],axis=1)   

    return new_data


# In[5]:


def get_data(data_type,file_num):


    if(data_type=='train'):
        data=pd.read_csv(f"Dataset/IHDP_a/ihdp_npci_train_{file_num}.csv")
    else:
        data = pd.read_csv(f"Dataset/IHDP_a/ihdp_npci_test_{file_num}.csv")

    x_data=pd.concat([data.iloc[:,0], data.iloc[:, 1:30]], axis = 1)
    x_data.iloc[:,18]=np.where(x_data.iloc[:,18]==2,1,0)
    #x_data_a=x_data.iloc[:,0:5]
    #x_data_b=x_data.iloc[:,5:30]
    #scaler.fit(x_data_b)
    #scaled_b = pd.DataFrame(scaler.fit_transform(x_data_b))
    #x_data=data.iloc[:, 5:30]
    #x_data_trans=pd.concat([x_data_a,scaled_b],axis=1)
    y_data_trans=data.iloc[:, 1]
    #y_data_trans=pd.DataFrame(scaler.fit_transform(data.iloc[:, 1].to_numpy().reshape(-1, 1)))
    #y_data_trans=y_data_trans.to_numpy().reshape(-1,)
    return x_data,y_data_trans


# In[6]:


class TarNet_VAE(nn.Module):
    def __init__(self,
                 input_dim,lat_dim_enc,
                 regularization):
        super(TarNet_VAE, self).__init__()
        self.encoder_gamma_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_gamma_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_gamma_3_mean = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_gamma_3_var = nn.Linear(lat_dim_enc, lat_dim_enc)

        self.encoder_delta_1 = nn.Linear(input_dim,hid_enc)
        self.encoder_delta_2 = nn.Linear(hid_enc, hid_enc)
        self.encoder_delta_mean = nn.Linear(hid_enc, lat_dim_enc)
        self.encoder_delta_var = nn.Linear(hid_enc, lat_dim_enc)

        self.encoder_upsilon_1 = nn.Linear(input_dim,hid_enc)
        self.encoder_upsilon_2 = nn.Linear(hid_enc, hid_enc)
        self.encoder_upsilon_mean = nn.Linear(hid_enc, lat_dim_enc)
        self.encoder_upsilon_var = nn.Linear(hid_enc, lat_dim_enc)


        """
        self.Irr_1 = nn.Linear(input_dim,hid_enc)
        self.Irr_2 = nn.Linear(hid_enc, hid_enc)
        self.Irr_mean = nn.Linear(hid_enc, lat_dim_enc)
        self.Irr_var = nn.Linear(hid_enc, lat_dim_enc)
        """
        #self.BN= nn.BatchNorm1d(lat_dim_enc)

        #self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.sig = nn.Sigmoid()


        #self.lambd = nn.Parameter(torch.tensor(1000.0)).cuda()



    def reparameterize(self, mean, logvar):
        #version 1
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

        #version 2
        #eps = torch.randn_like(logvar)
        #return mean + eps * logvar

        #version 3

        #return mean+logvar*self.N.sample(mean.shape)

    def forward(self, inputs):
        x_gamma = nn.functional.elu(self.encoder_gamma_1(inputs))
        x_gamma = nn.functional.elu(self.encoder_gamma_2(x_gamma))
        phi_gamma_mean = self.encoder_gamma_3_mean(x_gamma)
        phi_gamma_var = self.encoder_gamma_3_var(x_gamma)

        x_delta = nn.functional.elu(self.encoder_delta_1(inputs))
        x_delta = nn.functional.elu(self.encoder_delta_2(x_delta))
        phi_delta_mean = self.encoder_delta_mean(x_delta)
        phi_delta_var = self.encoder_delta_var(x_delta)

        x_upsilon = nn.functional.elu(self.encoder_upsilon_1(inputs))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_2(x_upsilon))
        phi_upsilon_mean = self.encoder_upsilon_mean(x_upsilon)
        phi_upsilon_var = self.encoder_upsilon_var(x_upsilon)

        """
        x_irr = nn.functional.elu(self.Irr_1(inputs))
        x_irr = nn.functional.elu(self.Irr_2(x_irr))
        phi_irr_mean = self.Irr_mean(x_irr)
        phi_irr_var = self.Irr_var(x_irr)
        """

        phi_gamma_z = self.reparameterize(phi_gamma_mean, phi_gamma_var)
        phi_delta_z = self.reparameterize(phi_delta_mean, phi_delta_var)
        phi_upsilon_z = self.reparameterize(phi_upsilon_mean, phi_delta_var)
        """
        phi_delta_z = self.reparameterize(phi_delta_mean, phi_delta_var)
        phi_upsilon_z = self.reparameterize(phi_upsilon_mean, phi_upsilon_var)
        phi_irr_z = self.reparameterize(phi_irr_mean, phi_irr_var)
        """

        #phi_gamma=phi_gamma_z[:,fstart:fend]
        #phi_delta=phi_gamma_z[:,sstart:send]
        #phi_upsilon=phi_gamma_z[:,tstart:tend]
        #phi_irr=phi_gamma_z[:,frstart:frend]

        #phi=torch.cat((phi_gamma,phi_delta,phi_upsilon_z,phi_irr),1)

        return phi_gamma_z,phi_delta_z,phi_upsilon_z,phi_gamma_mean,phi_gamma_var,phi_delta_mean,phi_delta_var,phi_upsilon_mean,phi_upsilon_var


class TarNet_DRCFR(nn.Module):
    def __init__(self,
                 input_dim,lat_dim_enc,
                 regularization):
        super(TarNet_DRCFR, self).__init__()
        self.encoder_gamma_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_gamma_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_gamma_3 = nn.Linear(lat_dim_enc, lat_dim_enc)


        self.encoder_delta_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_delta_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_delta_3 = nn.Linear(lat_dim_enc, lat_dim_enc)




        self.encoder_upsilon_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_upsilon_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_upsilon_3 = nn.Linear(lat_dim_enc, lat_dim_enc)


        self.sig = nn.Sigmoid()
        #self.BN= nn.BatchNorm1d(lat_dim_enc)






    def forward(self, inputs):
        x_gamma = nn.functional.elu(self.encoder_gamma_1(inputs))
        x_gamma = nn.functional.elu(self.encoder_gamma_2(x_gamma))

        """

        x_gamma = nn.functional.elu(self.encoder_gamma_3(x_gamma))
        x_gamma = nn.functional.elu(self.encoder_gamma_4(x_gamma))
        x_gamma = nn.functional.elu(self.encoder_gamma_5(x_gamma))
        x_gamma = nn.functional.elu(self.encoder_gamma_6(x_gamma))
        x_gamma = nn.functional.elu(self.encoder_gamma_7(x_gamma))
        """

        phi_gamma = self.encoder_gamma_3(x_gamma)

        x_delta = nn.functional.elu(self.encoder_delta_1(inputs))
        x_delta = nn.functional.elu(self.encoder_delta_2(x_delta))
        """

        x_delta = nn.functional.elu(self.encoder_delta_3(x_delta))
        x_delta = nn.functional.elu(self.encoder_delta_4(x_delta))
        x_delta = nn.functional.elu(self.encoder_delta_5(x_delta))
        x_delta = nn.functional.elu(self.encoder_delta_6(x_delta))
        x_delta = nn.functional.elu(self.encoder_delta_7(x_delta))
         """


        phi_delta = self.encoder_delta_3(x_delta)



        x_upsilon = nn.functional.elu(self.encoder_upsilon_1(inputs))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_2(x_upsilon))
        """
        x_upsilon = nn.functional.elu(self.encoder_upsilon_3(x_upsilon))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_4(x_upsilon))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_5(x_upsilon))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_6(x_upsilon))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_7(x_upsilon))
         """


        phi_upsilon = self.encoder_upsilon_3(x_upsilon)


        # Regressors
        #del_ups=torch.cat((phi_delta, phi_upsilon), 1)
        #out_y0 = self.regressor1_y0(del_ups)
        #out_y0 = self.regressor2_y0(out_y0)
        #y0 = self.regressorO_y0(out_y0)

        #out_y1 = self.regressor1_y1(del_ups)
        #out_y1 = self.regressor2_y1(out_y1)
        #y1 = self.regressorO_y1(out_y1)

        # classifires
        #gam_del=torch.cat((phi_gamma,phi_delta), 1)
        #out_w=self.classifier_w1(phi_delta)
        #out_w=self.classifier_w2(out_w)
        #out_w_f=self.sig(self.classifier_w3(out_w))

        #out_t=self.classifier_t1(gam_del)
        #out_t=self.classifier_t2(out_t)
        #out_t_f=self.sig(self.classifier_t3(out_t))

        # Returning arguments

        #concat = torch.cat((y0, y1), 1)
        return phi_gamma,phi_delta,phi_upsilon#out_w_f,out_t_f

class TarNet_DRI(nn.Module):
    def __init__(self,
                 input_dim,lat_dim_enc,
                 regularization):
        super(TarNet_DRI, self).__init__()
        self.encoder_gamma_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_gamma_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_gamma_3 = nn.Linear(lat_dim_enc, lat_dim_enc)

        self.encoder_delta_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_delta_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_delta_3 = nn.Linear(lat_dim_enc, lat_dim_enc)

        self.encoder_upsilon_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_upsilon_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_upsilon_3 = nn.Linear(lat_dim_enc, lat_dim_enc)

        self.Irr_1 = nn.Linear(input_dim,lat_dim_enc)
        self.Irr_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.Irr_3 = nn.Linear(lat_dim_enc, lat_dim_enc)

        self.sig = nn.Sigmoid()
        self.BN= nn.BatchNorm1d(lat_dim_enc)






    def forward(self, inputs):
        x_gamma = nn.functional.elu(self.encoder_gamma_1(inputs))
        x_gamma = nn.functional.elu(self.encoder_gamma_2(x_gamma))
        phi_gamma = self.encoder_gamma_3(x_gamma)

        x_delta = nn.functional.elu(self.encoder_delta_1(inputs))
        x_delta = nn.functional.elu(self.encoder_delta_2(x_delta))
        phi_delta = self.encoder_delta_3(x_delta)

        x_upsilon = nn.functional.elu(self.encoder_upsilon_1(inputs))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_2(x_upsilon))
        phi_upsilon = self.encoder_upsilon_3(x_upsilon)


        x_irr = nn.functional.elu(self.Irr_1(inputs))
        x_irr = nn.functional.elu(self.Irr_2(x_irr))
        phi_irr = self.Irr_3(x_irr)

        # Regressors
        #del_ups=torch.cat((phi_delta, phi_upsilon), 1)
        #out_y0 = self.regressor1_y0(del_ups)
        #out_y0 = self.regressor2_y0(out_y0)
        #y0 = self.regressorO_y0(out_y0)

        #out_y1 = self.regressor1_y1(del_ups)
        #out_y1 = self.regressor2_y1(out_y1)
        #y1 = self.regressorO_y1(out_y1)

        # classifires
        #gam_del=torch.cat((phi_gamma,phi_delta), 1)
        #out_w=self.classifier_w1(phi_delta)
        #out_w=self.classifier_w2(out_w)
        #out_w_f=self.sig(self.classifier_w3(out_w))

        #out_t=self.classifier_t1(gam_del)
        #out_t=self.classifier_t2(out_t)
        #out_t_f=self.sig(self.classifier_t3(out_t))

        # Returning arguments

        #concat = torch.cat((y0, y1), 1)
        return phi_gamma,phi_delta,phi_upsilon,phi_irr#out_w_f,out_t_f


# In[7]:


class TarNet(nn.Module):
    def __init__(self,
                 input_dim,hid_enc,lat_dim_enc,
                 regularization):
        super(TarNet, self).__init__()
        self.encoder_gamma_1 = nn.Linear(input_dim,hid_enc)
        self.encoder_gamma_2 = nn.Linear(hid_enc, hid_enc)
        self.encoder_gamma_3_mean = nn.Linear(hid_enc, lat_dim_enc)
        self.encoder_gamma_3_var = nn.Linear(hid_enc, lat_dim_enc)

        self.sig = nn.Sigmoid()
        self.BN= nn.BatchNorm1d(lat_dim_enc)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()



    def reparameterize(self, mean, logvar):
        #version 1
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

        #version 2
        #eps = torch.randn_like(logvar)
        #return mean + eps * logvar

        #version 3

        #return mean+logvar*self.N.sample(mean.shape)

    def forward(self, inputs):
        x_gamma = nn.functional.elu(self.encoder_gamma_1(inputs))
        x_gamma = nn.functional.elu(self.encoder_gamma_2(x_gamma))
        phi_gamma_mean = self.encoder_gamma_3_mean(x_gamma)
        phi_gamma_var = self.encoder_gamma_3_var(x_gamma)

        phi_gamma_z = self.reparameterize(phi_gamma_mean, phi_gamma_var)


        phi_gamma=phi_gamma_z[:,fstart:fend]
        phi_delta=phi_gamma_z[:,sstart:send]
        phi_upsilon=phi_gamma_z[:,tstart:tend]
        phi_irr=phi_gamma_z[:,frstart:frend]

        phi=torch.cat((phi_gamma,phi_delta,phi_upsilon,phi_irr),1)

        return (phi_gamma_z,phi_gamma_mean,phi_gamma_var)


# In[8]:


#write_metrics_latex(save_path=os.path.join(SAVE_DIR, "metrics_cheatsheet.tex"))


# In[9]:


import os, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt

# ===========================
# Imports & global config
# ===========================


# =========================================================
# NeurIPS-style plotting utils + per-panel save functions
# =========================================================
def set_neurips_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass


def make_embedding_dashboard(Z, T, Y, method_name="Method", view_name="plot-view",
                             random_state=42, k_mix=10, k_smooth=10,
                             figsize=(12, 12),  # a bit taller to fit the extra panel
                             bins_mix=20, bins_smooth=30, vmax_cov=None):

    # 2D embedding
    E, emb_name = embed_2d(Z, random_state=random_state)

    # Data for panels
    vspec = _variance_spectrum(Z)
    mix_vals = _knn_mixing_values(Z, T, k=k_mix)
    C = _cov_heatmap_data(Z)

    # NEW: neighborhood outcome smoothness
    smooth_scores, smooth_info = neighborhood_outcome_smoothness(
        Z, Y, k=k_smooth, standardize=True, metric="euclidean", include_self=False
    )

    # Figure layout: 4 rows now
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=4, ncols=3, hspace=0.35, wspace=0.25)

    # [1] 2D colored by T
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(E[:, 0], E[:, 1], c=T, cmap="coolwarm", s=8, alpha=0.85)
    ax1.set_title(f"{method_name} — {view_name}\n{emb_name} colored by T", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])
    cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_ticks([np.min(T), np.max(T)])
    cb1.set_label("T")

    # [2] 2D colored by Y
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(E[:, 0], E[:, 1], c=Y, cmap="viridis", s=8, alpha=0.85)
    ax2.set_title(f"{emb_name} colored by Y", fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])
    cb2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Y")

    # [3] Variance spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(np.arange(1, len(vspec) + 1), vspec, marker='o', linewidth=1)
    ax3.set_title("Variance spectrum", fontsize=11)
    ax3.set_xlabel("Component #"); ax3.set_ylabel("Eigenvalue")
    ax3.grid(alpha=0.3)

    # [4] kNN mixing histogram
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.hist(mix_vals, bins=bins_mix, range=(0, 1), edgecolor="black", alpha=0.85)
    ax4.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax4.set_title(f"kNN mixing (k={k_mix}) — fraction of opposite-group neighbors", fontsize=11)
    ax4.set_xlabel("Fraction opposite"); ax4.set_ylabel("Count")

    # [5] Covariance heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    im = ax5.imshow(C, aspect="auto", interpolation="nearest", cmap="coolwarm",
                    vmin=-abs(vmax_cov) if vmax_cov else None,
                    vmax=abs(vmax_cov) if vmax_cov else None)
    ax5.set_title("Covariance (on standardized Z)", fontsize=11)
    ax5.set_xlabel("Dims"); ax5.set_ylabel("Dims")
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    # [6] NEW — Neighborhood Outcome Smoothness histogram
    ax6 = fig.add_subplot(gs[2, :])
    if smooth_info["discrete_Y"]:
        ax6.hist(smooth_scores, bins=bins_smooth, range=(0, 1), edgecolor="black", alpha=0.85)
        ax6.set_xlabel("Outcome neighborhood impurity  [0 = smooth]")
    else:
        ax6.hist(smooth_scores, bins=bins_smooth, edgecolor="black", alpha=0.85)
        ax6.set_xlabel("Outcome neighbor variance  [lower = smoother]")
    ax6.set_title(f"Neighborhood Outcome Smoothness (k={smooth_info['k']})", fontsize=11)
    ax6.set_ylabel("Count")
    ax6.grid(alpha=0.25)

    # [7] Footer stats
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis("off")
    n, d = Z.shape
    smooth_type = "impurity" if smooth_info["discrete_Y"] else "variance"
    footer = (f"N={n}, d={d} | PR={participation_ratio(Z):.2f} | "
              f"kNN mixing mean={mix_vals.mean():.3f} (ideal 0.5) | "
              f"Smoothness mean={smooth_info['mean']:.3f} ({smooth_type})")
    ax7.text(0.01, 0.5, footer, va="center", ha="left", fontsize=11)

    fig.suptitle(f"Embedding Dashboard — {method_name} ({view_name})", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _avoid_xlabel_cut(ax, rotate_ticks=False, pad_bottom=0.14, tick_rot=0, tick_fs=9):
    """
    Add bottom margin so xlabel/ticks don't get cut; optionally rotate ticks.
    """
    # A small bottom margin so labels fit
    ax.figure.subplots_adjust(bottom=pad_bottom)

    if rotate_ticks:
        for lab in ax.get_xticklabels():
            lab.set_rotation(tick_rot)
            lab.set_ha("right")
            lab.set_fontsize(tick_fs)

    # A little extra data margin so vlines/text are inside the axes
    ax.margins(x=0.02)


def _save_fig(fig, basepath):
    png = basepath + ".png"
    #pdf = basepath + ".pdf"

    # Make sure layout is computed before saving:
    try:
        fig.canvas.draw()           # force a layout pass
    except Exception:
        pass

    # Slightly larger pad so labels aren’t clipped
    fig.savefig(png, bbox_inches="tight", pad_inches=0.06)
    #fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[Saved] {png}")

_CMAP_T = "coolwarm"
_CMAP_Y = "viridis"
_CMAP_COV = "coolwarm"

# ----- small helpers used by panels -----
def _variance_spectrum(Z, top_k=30):
    Z = np.asarray(Z, float)
    C = np.cov(Z, rowvar=False)
    vals = np.linalg.eigvalsh(C).clip(min=0)
    vals = np.sort(vals)[::-1]  # descending order
    return vals[:top_k] 

def _knn_mixing_values(Z, T, k=10):
    Z = np.asarray(Z, float); T = np.asarray(T).astype(int)
    nn = NearestNeighbors(n_neighbors=min(k+1, len(Z))).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    frac_other = (T[idx] != T[:, None]).mean(axis=1)
    return frac_other

def _cov_heatmap_data(Z):
    Z = np.asarray(Z, float)
    Zs = StandardScaler().fit_transform(Z)
    C = np.cov(Zs, rowvar=False)
    return C

# ===========================
# Embedding helper
# ===========================
def embed_2d(Z, random_state=42, n_neighbors=15, min_dist=0.1):
    Z = np.asarray(Z, float)
    Zs = StandardScaler().fit_transform(Z)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=random_state, metric="euclidean")
        E = reducer.fit_transform(Zs)
        return E, "UMAP"
    if HAS_TSNE:
        E = TSNE(n_components=2, init="pca", random_state=random_state, learning_rate="auto").fit_transform(Zs)
        return E, "t-SNE"
    E = PCA(n_components=2, random_state=random_state).fit_transform(Zs)
    return E, "PCA"

# ===========================
# Outcome smoothness
# ===========================
def _is_discrete_y(y, max_unique=20):
    y = np.asarray(y)
    return np.issubdtype(y.dtype, np.integer) or np.unique(y).size <= max_unique

def neighborhood_outcome_smoothness(Z, Y, k=10, standardize=True, metric="euclidean", include_self=False):
    Z = np.asarray(Z, float)
    y = np.asarray(Y)
    if standardize:
        Z = StandardScaler().fit_transform(Z)
    n_neighbors = k if include_self else min(k + 1, len(Z))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)
    if not include_self:
        idx = idx[:, 1:]
    discrete = _is_discrete_y(y)
    scores = np.zeros(len(Z), dtype=float)
    if discrete:
        for i in range(len(Z)):
            neigh_y = y[idx[i]]
            _, counts = np.unique(neigh_y, return_counts=True)
            p_max = counts.max() / counts.sum()
            scores[i] = 1.0 - p_max  # 0 = smooth
    else:
        for i in range(len(Z)):
            neigh_y = y[idx[i]].astype(float)
            scores[i] = float(np.var(neigh_y, ddof=1)) if len(neigh_y) > 1 else 0.0
    info = {
        "mean": float(np.mean(scores)), "median": float(np.median(scores)),
        "std": float(np.std(scores)), "min": float(np.min(scores)), "max": float(np.max(scores)),
        "k": int(k), "discrete_Y": bool(discrete), "metric": metric, "include_self": bool(include_self),
    }
    return scores, info

# ===========================
# Per-panel: single figures
# ===========================
def save_embed_by_T(Z, T, out_base, random_state=42, figsize=(3.4, 3.4)):
    set_neurips_style()
    E, emb_name = embed_2d(Z, random_state=random_state)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    sc = ax.scatter(E[:, 0], E[:, 1], c=T, cmap=_CMAP_T, s=8, alpha=0.9, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{emb_name}: colored by $T$")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$T$")
    _save_fig(fig, out_base + "_embed_T")

def save_embed_by_Y(Z, Y, out_base, random_state=42, figsize=(3.4, 3.4)):
    set_neurips_style()
    E, emb_name = embed_2d(Z, random_state=random_state)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    sc = ax.scatter(E[:, 0], E[:, 1], c=Y, cmap=_CMAP_Y, s=8, alpha=0.9, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{emb_name}: colored by $Y$")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$Y$")
    _save_fig(fig, out_base + "_embed_Y")

def save_variance_spectrum(Z, out_base, figsize=(3.6, 3.2)):
    set_neurips_style()
    vals = _variance_spectrum(Z)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(vals)+1), vals, marker='o', linewidth=1)
    ax.set_xlabel("component #"); ax.set_ylabel("eigenvalue")
    ax.set_title("Variance spectrum")
    _save_fig(fig, out_base + "_variance_spectrum")


def save_knn_mixing_hist(Z, T, k=10, out_base=None, figsize=(3.6, 3.2)):
    set_neurips_style()
    mix_vals = _knn_mixing_values(Z, T, k=k)
    mean_mix = float(np.mean(mix_vals))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.hist(mix_vals, bins=24, range=(0, 1), edgecolor="black", alpha=0.85)
    # target 0.5 and empirical mean
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="target 0.5")
    ax.axvline(mean_mix, color="tab:orange", linestyle="-", linewidth=1.2, label=f"mean = {mean_mix:.3f}")
    ax.text(mean_mix, ax.get_ylim()[1]*0.92, f"{mean_mix:.3f}",
        ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
    ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=9)

    ax.set_xlabel("fraction of opposite-group neighbors")
    ax.set_ylabel("count")
    ax.set_title(f"kNN mixing (k={k})")
    _avoid_xlabel_cut(ax, rotate_ticks=False, pad_bottom=0.14)
    _save_fig(fig, out_base + "_knn_mixing")
    print(f"[kNN mixing] mean={mean_mix:.6f}, std={np.std(mix_vals):.6f}")


def save_cov_heatmap(Z, out_base, vmax=None, figsize=(3.6, 3.2)):
    set_neurips_style()
    Zs = StandardScaler().fit_transform(Z)
    C = np.cov(Zs, rowvar=False)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    im = ax.imshow(C, aspect="auto", interpolation="nearest",
                   cmap=_CMAP_COV,
                   vmin=-abs(vmax) if vmax else None,
                   vmax= abs(vmax) if vmax else None)
    ax.set_xlabel("dims"); ax.set_ylabel("dims")
    ax.set_title("Covariance (z-scored)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_base + "_cov_heatmap")

def save_outcome_smoothness_hist(Z, Y, k=10, out_base=None, figsize=(3.6, 3.2)):
    set_neurips_style()
    scores, info = neighborhood_outcome_smoothness(
        Z, Y, k=k, standardize=True, metric="euclidean", include_self=False
    )
    mean_smooth = info["mean"]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    if info["discrete_Y"]:
        ax.hist(scores, bins=30, range=(0, 1), edgecolor="black", alpha=0.85)
        ax.set_xlabel("neighborhood impurity  (0 = smooth)")
    else:
        ax.hist(scores, bins=30, edgecolor="black", alpha=0.85)
        ax.set_xlabel("neighborhood outcome variance  (lower = smoother)")

    # empirical mean
    ax.axvline(mean_smooth, color="tab:orange", linestyle="-", linewidth=1.2, label=f"mean = {mean_smooth:.3f}")
    ax.text(mean_smooth, ax.get_ylim()[1]*0.92, f"{mean_smooth:.3f}",
        ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    ax.set_ylabel("count")
    ax.set_title(f"Outcome smoothness (k={info['k']})")
    _avoid_xlabel_cut(ax, rotate_ticks=False, pad_bottom=0.14)
    _save_fig(fig, out_base + "_outcome_smoothness")
    print(f"[Outcome smoothness] mean={mean_smooth:.6f}, std={info['std']:.6f}, discrete_Y={info['discrete_Y']}")


def save_all_panels_per_space(Z, T, Y, space_name, view_name, out_dir,
                              *, k_mix=10, k_smooth=10, seed=42):
    """
    Saves 6 separate NeurIPS-style figures for a given space (X or Z-view)
    """
    _ensure_dir(out_dir)
    base = os.path.join(out_dir, f"{space_name}__{view_name}")
    save_embed_by_T(Z, T, base, random_state=seed)
    save_embed_by_Y(Z, Y, base, random_state=seed)
    save_variance_spectrum(Z, base)
    save_knn_mixing_hist(Z, T, k=k_mix, out_base=base)
    save_cov_heatmap(Z, base)
    save_outcome_smoothness_hist(Z, Y, k=k_smooth, out_base=base)

# ===========================
# Neighborhood preservation & density
# ===========================
def _knn_indices(X, k, metric="euclidean", standardize=False):
    X = np.asarray(X, float)
    if standardize:
        X = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(k+1, len(X)), metric=metric)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    return idx[:, 1:]  # drop self

def _rank_matrix(X, metric="euclidean", standardize=False):
    X = np.asarray(X, float)
    n = len(X)
    idx = _knn_indices(X, k=n-1, metric=metric, standardize=standardize)  # full ranking
    R = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        R[i, idx[i]] = np.arange(1, n, dtype=np.int32)  # 1..n-1
    return R, idx

def trustworthiness_continuity(
    X_high, X_low, k=10, metric_high="euclidean", metric_low="euclidean",
    standardize_high=True, standardize_low=True
):
    """
    Trustworthiness: are low-d neighbors also neighbors in high-d?
    Continuity:      are high-d neighbors preserved in low-d?
    Both in [0,1]; higher is better.
    """
    X_high = np.asarray(X_high, float); X_low = np.asarray(X_low, float)
    n = len(X_high); k = int(min(k, n-2))

    R_high, _ = _rank_matrix(X_high, metric=metric_high, standardize=standardize_high)
    R_low,  _ = _rank_matrix(X_low,  metric=metric_low,  standardize=standardize_low)
    N_high = _knn_indices(X_high, k=k, metric=metric_high, standardize=standardize_high)
    N_low  = _knn_indices(X_low,  k=k, metric=metric_low,  standardize=standardize_low)

    pen_trust, pen_cont = 0.0, 0.0
    for i in range(n):
        U = set(N_low[i]).difference(set(N_high[i]))
        if U: pen_trust += np.sum(R_high[i][list(U)] - k)
        V = set(N_high[i]).difference(set(N_low[i]))
        if V: pen_cont  += np.sum(R_low[i][list(V)] - k)

    norm = 2.0 / (n * k * (2*n - 3*k - 1))
    trust = float(np.clip(1.0 - norm * pen_trust, 0.0, 1.0))
    cont  = float(np.clip(1.0 - norm * pen_cont,  0.0, 1.0))
    return trust, cont

def neighborhood_density_scores(Z, k=10, metric="euclidean", standardize=True, mode="kth"):
    """
    mode='kth': distance to k-th NN (robust radius); mode='mean': mean distance to k NNs.
    """
    Z = np.asarray(Z, float)
    if standardize:
        Z = StandardScaler().fit_transform(Z)
    nn = NearestNeighbors(n_neighbors=min(k+1, len(Z)), metric=metric).fit(Z)
    dists, _ = nn.kneighbors(Z, return_distance=True)
    d = dists[:, 1:]  # drop self
    scores = d[:, -1] if mode == "kth" else d.mean(axis=1)
    info = {
        "k": int(k), "mode": mode, "mean": float(scores.mean()),
        "median": float(np.median(scores)), "std": float(scores.std()),
        "min": float(scores.min()), "max": float(scores.max())
    }
    return scores, info

def plot_density_hist(scores, bins=30, title="Neighborhood Density (lower = denser)"):
    plt.hist(scores, bins=bins, edgecolor="black", alpha=0.85)
    plt.xlabel("local density score"); plt.ylabel("count")
    plt.title(title); plt.grid(alpha=0.25)
def wasserstein_sliced(Z1, Z0, n_projections=256, rng=None, standardize=True):
    rng = default_rng(42) if rng is None else rng
    Z1 = np.asarray(Z1, float); Z0 = np.asarray(Z0, float)
    if standardize:
        mu = np.mean(np.vstack([Z1, Z0]), axis=0)
        sd = np.std(np.vstack([Z1, Z0]), axis=0, ddof=1) + 1e-12
        Z1 = (Z1 - mu) / sd; Z0 = (Z0 - mu) / sd
    d = Z1.shape[1]; ws = []
    for _ in range(n_projections):
        v = rng.normal(size=d); v /= np.linalg.norm(v) + 1e-12
        a = np.sort(Z1 @ v); b = np.sort(Z0 @ v)
        m = min(len(a), len(b))
        ws.append(np.mean(np.abs(a[:m] - b[:m])))
    return float(np.mean(ws))

def plot_density_scatter_2d(E2, scores, title="2D embedding colored by density"):
    inv = (scores.max() - scores) if scores.max() > scores.min() else (1.0 - scores)
    sc = plt.scatter(E2[:, 0], E2[:, 1], c=inv, s=8, alpha=0.9, cmap="viridis")
    plt.colorbar(sc, fraction=0.046, pad=0.04, label="relative density (higher = denser)")
    plt.title(title); plt.xticks([]); plt.yticks([])

# ========= NEW: single combined QC panel (two methods) =========
# ========= Split QC panels: save each figure separately =========
def save_qc_panel_for_method(
    X_pool, Z_view, method_name, out_dir, *, k_tc=10, k_density=10, seed=42,
    save_combined=False  # set True if you ALSO want the old combined page
):
    """
    Saves separate PNGs for:
      - Trust & Continuity bar  -> QC_{method}_bar.png
      - Density histogram       -> QC_{method}_hist.png
      - 2D density scatter      -> QC_{method}_scatter.png
      - Stats text card         -> QC_{method}_stats.png

    If save_combined=True, also saves the original combined page as QC_{method}.png
    """
    set_neurips_style()
    _ensure_dir(out_dir)

    # ---- metrics ----
    trust, cont = trustworthiness_continuity(X_pool, Z_view, k=k_tc)
    dens, dinfo = neighborhood_density_scores(Z_view, k=k_density, mode="kth")
    E, emb_name = embed_2d(Z_view, random_state=seed)

    out_base = os.path.join(out_dir, f"QC_{method_name}")

    # ---- [A] Trust & Continuity bar ----
    fig = plt.figure(figsize=(4.2, 3.3), constrained_layout=True)
    axA = fig.add_subplot(111)
    vals = [trust, cont]; labels = ["Trustworthiness", "Continuity"]
    bars = axA.bar(labels, vals, alpha=0.9)
    axA.tick_params(axis='x', labelsize=8); axA.tick_params(axis='y', labelsize=8)
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("score", fontsize=9)
    axA.set_title(f"Neighbor preservation (k={k_tc})", fontsize=9)
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
    _save_fig(fig, out_base + "_bar")

    # ---- [B] Density histogram ----
    fig = plt.figure(figsize=(4.2, 3.3), constrained_layout=True)
    axB = fig.add_subplot(111)
    axB.hist(dens, bins=30, edgecolor="black", alpha=0.85)
    axB.set_title(f"Neighborhood density (k={k_density})", fontsize=9)
    axB.set_xlabel("k-NN radius (lower = denser)", fontsize=9)
    axB.set_ylabel("count", fontsize=9)
    _save_fig(fig, out_base + "_hist")

    # ---- [C] 2D scatter colored by density ----
    fig = plt.figure(figsize=(4.2, 3.3), constrained_layout=True)
    axC = fig.add_subplot(111)
    inv = (dens.max() - dens) if dens.max() > dens.min() else (1.0 - dens)
    sc = axC.scatter(E[:, 0], E[:, 1], c=inv, s=8, alpha=0.95, cmap="viridis")
    axC.set_xticks([]); axC.set_yticks([])
    axC.set_title(f"{method_name} — {emb_name} (colored by density)", fontsize=9)
    cbar = plt.colorbar(sc, ax=axC, fraction=0.046, pad=0.04)
    cbar.set_label("relative density (higher = denser)")
    _save_fig(fig, out_base + "_scatter")

    # ---- [D] Stats card ----
    fig = plt.figure(figsize=(4.2, 2.1), constrained_layout=True)
    axD = fig.add_subplot(111); axD.axis("off")
    title = f"QC — {method_name}"
    txt = (f"Trust={trust:.3f}, Continuity={cont:.3f}\n"
           f"Density: mean={dinfo['mean']:.4f}, med={dinfo['median']:.4f}, std={dinfo['std']:.4f}")
    axD.text(0.01, 0.75, title, fontsize=10, weight="bold")
    axD.text(0.01, 0.28, txt, fontsize=9)
    _save_fig(fig, out_base + "_stats")

    # ---- (optional) also save the original combined page ----
    if save_combined:
        fig = plt.figure(figsize=(8.8, 4.8), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1, 1.2], width_ratios=[1.1, 1.1, 1.2])

        axA = fig.add_subplot(gs[0, 0])
        vals = [trust, cont]; labels = ["Trustworthiness", "Continuity"]
        bars = axA.bar(labels, vals, alpha=0.9)
        axA.tick_params(axis='x', labelsize=6); axA.tick_params(axis='y', labelsize=6)
        axA.set_ylim(0, 1.0); axA.set_ylabel("score"); axA.set_title(f"Neighbor preservation (k={k_tc})")
        for b, v in zip(bars, vals):
            axA.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        axB = fig.add_subplot(gs[0, 1])
        axB.hist(dens, bins=30, edgecolor="black", alpha=0.85)
        axB.set_title(f"Neighborhood density (k={k_density})")
        axB.set_xlabel("k-NN radius (lower = denser)"); axB.set_ylabel("count")

        axC = fig.add_subplot(gs[:, 2])
        sc = axC.scatter(E[:, 0], E[:, 1], c=inv, s=8, alpha=0.95, cmap="viridis")
        axC.set_xticks([]); axC.set_yticks([])
        axC.set_title(f"{method_name} — {emb_name} (colored by density)")
        cbar = plt.colorbar(sc, ax=axC, fraction=0.046, pad=0.04)
        cbar.set_label("relative density (higher = denser)")

        axD = fig.add_subplot(gs[1, 0]); axD.axis("off")
        axD.text(0.0, 0.5,
                 f"Trust={trust:.3f}, Continuity={cont:.3f} | "
                 f"Density: mean={dinfo['mean']:.4f}, med={dinfo['median']:.4f}, std={dinfo['std']:.4f}",
                 ha="left", va="center", fontsize=10)

        _save_fig(fig, out_base)
# ====================================
# Space-level eval (single matrix)
# ====================================
def eval_space(X, T, *, label, n_proj=256):
    X = np.asarray(X, float); T = np.asarray(T, int)
    X1, X0 = X[T == 1], X[T == 0]
    rng = default_rng(42)
    out = {
        "space": label,
        "MMD2": mmd2_rbf(X1, X0),
        "SW1": wasserstein_sliced(X1, X0, n_projections=n_proj, rng=rng, standardize=True),
        "kNN_mixing": knn_mixing(X, T, k=10),
        "AUC_T_given": treatment_auc(X, T),
        "PR": participation_ratio(X),
        "CH_ratio": ch_ratio_two_groups(X, T),
    }
    return out

# ====================================
# Component routing
# ====================================
def get_view(components: dict, selector):
    if isinstance(selector, str):
        if selector == "all":
            parts = [components[k] for k in sorted(components.keys())]
        elif "+" in selector:
            keys = [s.strip() for s in selector.split("+")]
            parts = [components[k] for k in keys]
        else:
            parts = [components[selector]]
    else:
        parts = [components[k] for k in selector]
    return np.concatenate(parts, axis=1)

VIEW_POLICY = {
    "DRCFR": {
        "plot": "delta+upsilon", "T_auc": "delta+upsilon",
        "Y_sufficiency": "delta+upsilon", "PR": "delta+upsilon",
        "CH_ratio": "delta+upsilon", "Active_units": "delta+upsilon",
        "kNN_mixing": "delta+upsilon", "SW1": "delta+upsilon", "MMD2": "delta+upsilon",
    },
    "VAE": {
        "plot": "delta+upsilon", "T_auc": "delta+upsilon",
        "Y_sufficiency": "delta+upsilon", "PR": "delta+upsilon",
        "CH_ratio": "delta+upsilon", "Active_units": "delta+upsilon",
        "kNN_mixing": "delta+upsilon", "SW1": "delta+upsilon", "MMD2": "delta+upsilon",
    },
    "Ours": {
        "plot": "delta+upsilon", "T_auc": "delta+upsilon",
        "Y_sufficiency": "delta+upsilon", "PR": "delta+upsilon",
        "CH_ratio": "delta+upsilon", "Active_units": "delta+upsilon",
        "kNN_mixing": "delta+upsilon", "SW1": "delta+upsilon", "MMD2": "delta+upsilon",
    },
    "CFR": {
        "plot": "all", "T_auc": "all", "Y_sufficiency": "all", "PR": "all",
        "CH_ratio": "all", "Active_units": "all", "kNN_mixing": "all", "SW1": "all", "MMD2": "all",
    },
}

# ====================================
# Per-method evaluation
# ====================================
def eval_method_with_views(method_name, comps, T, Y, X_mi_ref_abs=None, H_y=None, *, n_proj=256, seed=42):
    r = {"space": f"Z:{method_name}"}
    pol = VIEW_POLICY[method_name]
    Z_pr = get_view(comps, pol["PR"]);           r["PR"] = participation_ratio(Z_pr)
    Z_ch = get_view(comps, pol["CH_ratio"]);     r["CH_ratio"] = ch_ratio_two_groups(Z_ch, T)
    Z_mix = get_view(comps, pol["kNN_mixing"]);  r["kNN_mixing"] = knn_mixing(Z_mix, T, k=10)
    Z_sw = get_view(comps, pol["SW1"]);          Z_mmd = get_view(comps, pol["MMD2"])
    Z_sw1, Z_sw0 = Z_sw[T == 1], Z_sw[T == 0];   Z_m1, Z_m0  = Z_mmd[T == 1], Z_mmd[T == 0]
    rng = default_rng(42)
    r["SW1"]  = wasserstein_sliced(Z_sw1, Z_sw0, n_projections=n_proj, rng=rng, standardize=True)
    r["MMD2"] = mmd2_rbf(Z_m1, Z_m0)
    Z_t = get_view(comps, pol["T_auc"]);         r["AUC_T_given"] = treatment_auc(Z_t, T)
    Z_y = get_view(comps, pol["Y_sufficiency"])
    suff = outcome_sufficiency_internal_mi(Z_y, Y, cv_splits=5, seed=seed, alphas=(0.1,1.0,10.0), mi_k=5)
    r["R2"] = suff["R2"]; r["RMSE"] = suff["RMSE"]
    if H_y is not None: r["MI_ratio"] = float(suff["_MI_abs"] / max(H_y, 1e-12))
    elif X_mi_ref_abs is not None: r["MI_retention"] = float(suff["_MI_abs"] / max(X_mi_ref_abs, 1e-12))
    return r

# ====================================
# Model loading + per-file processing
# (YOU must provide: TarNet classes, get_data, add_dummy_features_shuffle, scaler_x, load_masks, load_rep)
# ====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(input_features, lat_dim, hid_enc, file_no, dummies):
    Enc_VAE = TarNet_VAE(input_features, lat_dim, .1).to(device)
    Enc_VAE.load_state_dict(torch.load(f"PR_ana/VAE_IHDP_{file_no}_{dummies}.pth", map_location=device))
    Enc_VAE.eval()
    Enc_DRCFR = TarNet_DRCFR(input_features, lat_dim, .1).to(device)
    Enc_DRCFR.load_state_dict(torch.load(f"PR_ana/DRCFR_IHDP_{file_no}_{dummies}.pth", map_location=device))
    Enc_DRCFR.eval()
    Enc = TarNet_DRI(input_features, hid_enc, .1).to(device)
    Enc.load_state_dict(torch.load(f"PR_ana/DRI_ITE_IHDP_{file_no}_{dummies}.pth", map_location=device))
    Enc.eval()
    return Enc, Enc_DRCFR, Enc_VAE

def load_rep(file_no, out_dir="introduction_to_CFR-main/saved_reps/", device="cuda:0"):
    path = f"{out_dir}/IHDP_rep_{file_no}_{dummies}.pt"
    data = torch.load(path, map_location=device)
    return {k: v.to(device) for k, v in data.items()}
def knn_mixing(Z, T, k=10):
    Z = np.asarray(Z, float); T = np.asarray(T, int)
    nn = NearestNeighbors(n_neighbors=min(k+1, len(Z))).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    return float((T[idx] != T[:, None]).mean())

@torch.no_grad()
def process_one_file(file_no, *, input_features, lat_dim, hid_enc, scaler_x, dummies):
    Enc, Enc_DRCFR, Enc_VAE = load_models(input_features, lat_dim, hid_enc, file_no, dummies)

    x_data_tr, y_data_tr = get_data('train', file_no)
    x_data_n = add_dummy_features_shuffle(x_data_tr, dummies)
    x_data_n_nor = pd.DataFrame(scaler_x.fit_transform(x_data_n.iloc[:, 5:]))
    data_train_tran = pd.concat([x_data_n.iloc[:, 0:5], x_data_n_nor], axis=1)

    data_np = data_train_tran.to_numpy()
    data_t = torch.from_numpy(data_np.astype(np.float32)).to(device)
    X_t = data_t[:, 5:]; X_np = data_np[:, 5:]; T_np = data_np[:, 0]; Y_np = data_np[:, 1]

    go, do, uo,oo = Enc(X_t)                       # Ours



    # 3) Embeddings (components)
    """
    phi, phi_mean, phi_var = Enc(X_t)                       # Ours
    masks = load_masks(file_no)
    mg=masks["mask_gamma"]
    md=masks["mask_delta"]
    mu=masks["mask_upsilon"]
    mo=masks["mask_omega"]


    mg=( mg>0.5).float()
    md = ( md>0.5).float()
    mu = ( mu>0.5).float()
    mo = ( mo>0.5).float()

    go=phi*mg
    do=phi*md
    uo=phi*mu
    """

    """
    mg = masks["mask_gamma"]
    md = masks["mask_delta"]
    mu = masks["mask_upsilon"]
    go = phi*mg; do = phi*md; uo = phi*mu
    """
    # DRCFR & VAE
    gD, dD, uD = Enc_DRCFR(X_t)
    gV, dV, uV, muV, lvV,phi_delta_mean,phi_delta_var,phi_upsilon_mean,phi_upsilon_var= Enc_VAE(X_t)

    # CFR (single rep)
    repre = load_rep(file_no)
    CFR_rep = repre["rep"]  # tensor (N,d_cfr)

    Z_components = {
        "Ours":  {
                  "gamma": go.detach().cpu().numpy(),
                  "delta": do.detach().cpu().numpy(),
                  "upsilon": uo.detach().cpu().numpy()},
        "DRCFR": {"gamma": gD.detach().cpu().numpy(),
                  "delta": dD.detach().cpu().numpy(),
                  "upsilon": uD.detach().cpu().numpy()},
        "VAE":   {"gamma": gV.detach().cpu().numpy(),
                  "delta": dV.detach().cpu().numpy(),
                  "upsilon": uV.detach().cpu().numpy()},
        "CFR":   {"all": CFR_rep.detach().cpu().numpy()}
    }
    """
    encoder_stats = {
        "Ours": {"mu": phi_mean.detach().cpu().numpy(), "logvar": phi_var.detach().cpu().numpy()},
        "VAE":  {"mu": muV.detach().cpu().numpy(),      "logvar": lvV.detach().cpu().numpy()},
    }
    """

    rX = eval_space(X_np, T_np, label="X", n_proj=256)
    suff_X = outcome_sufficiency_internal_mi(X_np, Y_np, cv_splits=5, seed=42, alphas=(0.1,1.0,10.0), mi_k=5)
    rX["R2"] = suff_X["R2"]; rX["RMSE"] = suff_X["RMSE"]

    X_mi_ref_abs = None; H_y = None
    if _is_discrete_y(Y_np):
        H_y = suff_X.get("_H_y", _entropy_discrete(Y_np))
        rX["MI_ratio"] = float(suff_X["_MI_abs"] / max(H_y, 1e-12))
    else:
        X_mi_ref_abs = suff_X["_MI_abs"]; rX["MI_retention"] = 1.0

    rows = [rX]
    for method_name, comps in Z_components.items():
        r = eval_method_with_views(method_name, comps, T_np, Y_np,
                                   X_mi_ref_abs=X_mi_ref_abs, H_y=H_y,
                                   n_proj=256, seed=42)
        # Active units (if available)
        """
        r["Active_units"] = np.nan
        if method_name in encoder_stats and all(k in encoder_stats[method_name] for k in ("mu","logvar")):
            n_active = (0.5 * (encoder_stats[method_name]["mu"]**2 + np.exp(encoder_stats[method_name]["logvar"])
                               - 1.0 - encoder_stats[method_name]["logvar"])).mean(0)
            r["Active_units"] = int((n_active > 1e-3).sum())
        """
        rows.append(r)

    pooled = {"X": X_np, "T": T_np, "Y": Y_np, "Z_components": Z_components}
    return rows, pooled

def participation_ratio(Z):
    Z = np.asarray(Z, float)
    if Z.ndim != 2 or Z.shape[0] < 2: return np.nan
    C = np.cov(Z, rowvar=False)
    w = np.linalg.eigvalsh(C).clip(min=0)
    s1, s2 = w.sum(), (w**2).sum() + 1e-12
    return float((s1**2) / s2)
def qc_plot_all_methods(X_pool, Z_components_pool, out_dir="rep_eval_out/qc",
                        view_key="plot", k_tc=10, k_density=10, seed=42):
   def qc_plot_all_methods_separate(X_pool, Z_components_pool, out_dir="rep_eval_out/qc_sep",
                                 view_key="plot", k_tc=10, k_density=10, seed=42):
    """
    For each method (and original X), compute metrics once and save
    THREE separate figures:
      1) 2D scatter colored by density
      2) Density histogram
      3) Stats card (trust/cont + density summary text)
    """
    os.makedirs(out_dir, exist_ok=True)

    def _save_scatter(E, dens, title, out_path):
        plt.figure(figsize=(5.0, 4.0), constrained_layout=True)
        inv = (dens.max() - dens) if dens.max() > dens.min() else (1.0 - dens)
        sc = plt.scatter(E[:, 0], E[:, 1], c=inv, s=8, alpha=0.95, cmap="viridis")
        plt.xticks([]); plt.yticks([])
        plt.title(title)
        cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
        cbar.set_label("relative density (higher = denser)")
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[Saved] {out_path}")

    def _save_hist(dens, k, title, out_path):
        plt.figure(figsize=(5.0, 4.0), constrained_layout=True)
        plt.hist(dens, bins=30, edgecolor="black", alpha=0.85)
        plt.xlabel("k-NN radius (lower = denser)")
        plt.ylabel("count")
        plt.title(f"{title} (k={k})")
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[Saved] {out_path}")

    def _save_stats_card(text, title, out_path):
        plt.figure(figsize=(5.0, 2.2), constrained_layout=True)
        ax = plt.gca(); ax.axis("off")
        ax.text(0.01, 0.7, title, fontsize=12, weight="bold")
        ax.text(0.01, 0.28, text, fontsize=10)
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[Saved] {out_path}")

    # ---------- Original space X ----------
    trust_x, cont_x = trustworthiness_continuity(X_pool, X_pool, k=k_tc)
    dens_x, dinfo_x = neighborhood_density_scores(X_pool, k=k_density, mode="kth")
    E_x, emb_name_x = embed_2d(X_pool, random_state=seed)

    base_x = os.path.join(out_dir, "X")
    _save_scatter(E_x, dens_x, title=f"X — {emb_name_x} (colored by density)",
                  out_path=base_x + "_scatter.png")
    _save_hist(dens_x, k_density, title="X density",
               out_path=base_x + "_hist.png")
    _save_stats_card(
        text=(f"Neighborhood Preservation vs X:  Trust={trust_x:.3f},  Continuity={cont_x:.3f}\n"
              f"Density stats:  mean={dinfo_x['mean']:.4f},  median={dinfo_x['median']:.4f},  std={dinfo_x['std']:.4f}"),
        title="General QC — Original Space X",
        out_path=base_x + "_stats.png"
    )



    # ---------- Each learned method ----------
    for method_name, comps_list in Z_components_pool.items():
        # Merge components across files
        merged = {}
        for comps in comps_list:
            for k, arr in comps.items():
                merged.setdefault(k, []).append(arr)
        merged = {k: np.vstack(v) for k, v in merged.items()}

        view = VIEW_POLICY[method_name][view_key]
        Z_view = get_view(merged, view)

        trust, cont = trustworthiness_continuity(X_pool, Z_view, k=k_tc)
        dens, dinfo = neighborhood_density_scores(Z_view, k=k_density, mode="kth")
        E, emb_name = embed_2d(Z_view, random_state=seed)

        base = os.path.join(out_dir, f"{method_name}__{view.replace('+','_')}")
        _save_scatter(E, dens, title=f"{method_name} — {view} — {emb_name} (colored by density)",
                      out_path=base + "_scatter.png")
        _save_hist(dens, k_density, title=f"{method_name} density",
                   out_path=base + "_hist.png")
        _save_stats_card(
            text=(f"Neighborhood Preservation vs X:  Trust={trust:.3f},  Continuity={cont:.3f}\n"
                  f"Density stats:  mean={dinfo['mean']:.4f},  median={dinfo['median']:.4f},  std={dinfo['std']:.4f}"),
            title=f"General QC — {method_name} [{view}]",
            out_path=base + "_stats.png"
        )

def treatment_auc(Z, T):
    Z = np.asarray(Z, float); T = np.asarray(T, int)
    if len(np.unique(T)) < 2: return np.nan
    clf = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=1, random_state=42)
    clf.fit(Z, T)
    p = clf.predict_proba(Z)[:, 1]
    return float(roc_auc_score(T, p))

def ch_ratio_two_groups(Z, T):
    """Calinski–Harabasz-style ratio: between-centroid sep / within spread (2 groups)."""
    Z = np.asarray(Z, float); T = np.asarray(T).ravel()
    labels = np.unique(T)
    if labels.size != 2:
        return np.nan
    Z0, Z1 = Z[T == labels[0]], Z[T == labels[1]]
    if len(Z0) < 2 or len(Z1) < 2:
        return np.nan
    c0, c1 = Z0.mean(0), Z1.mean(0)
    sep2 = float(np.sum((c1 - c0)**2))
    r2_0 = float(np.mean(np.sum((Z0 - c0)**2, axis=1)))
    r2_1 = float(np.mean(np.sum((Z1 - c1)**2, axis=1)))
    denom = r2_0 + r2_1
    return float(sep2 / max(denom, 1e-12))
# ====================================
# Aggregate over files
# ====================================
def _agg_mean_std(rows, drop=("space","file_no")):
    out = {"space": rows[0]["space"]}
    keys = set().union(*rows) - set(drop)
    for k in sorted(keys):
        vs = [r[k] for r in rows if isinstance(r.get(k, np.nan), (int,float,np.floating)) and np.isfinite(r.get(k, np.nan))]
        out[k] = "-" if not vs else (f"{np.mean(vs):.4f}±{np.std(vs):.4f}" if k != "Active_units" else f"{np.mean(vs):.1f}±{np.std(vs):.1f}")
    return out

def _pairwise_sq_dists(A, B):
    A2 = (A**2).sum(1)[:, None]
    B2 = (B**2).sum(1)[None, :]
    return A2 + B2 - 2 * A @ B.T

def _median_heuristic_sigma(Z1, Z0):
    D11 = _pairwise_sq_dists(Z1, Z1)
    D00 = _pairwise_sq_dists(Z0, Z0)
    D10 = _pairwise_sq_dists(Z1, Z0)
    tri11 = D11[np.triu_indices_from(D11, 1)]
    tri00 = D00[np.triu_indices_from(D00, 1)]
    flat = np.concatenate([tri11, tri00, D10.ravel()])
    flat = flat[flat > 0]
    if flat.size == 0:
        return 1.0
    return np.sqrt(0.5 * np.median(flat))
def mmd2_rbf(Z1, Z0, sigma=None):
    Z1 = np.asarray(Z1, float); Z0 = np.asarray(Z0, float)
    m, n = len(Z1), len(Z0)
    if m < 2 or n < 2: return np.nan
    if sigma is None:
        sigma = _median_heuristic_sigma(Z1, Z0)
    K11 = np.exp(-_pairwise_sq_dists(Z1, Z1) / (2 * sigma**2))
    K00 = np.exp(-_pairwise_sq_dists(Z0, Z0) / (2 * sigma**2))
    K10 = np.exp(-_pairwise_sq_dists(Z1, Z0) / (2 * sigma**2))
    np.fill_diagonal(K11, 0.0); np.fill_diagonal(K00, 0.0)
    term11 = K11.sum() / (m * (m - 1))
    term00 = K00.sum() / (n * (n - 1))
    term10 = K10.mean()
    return term11 - 2 * term10 + term00

def _is_discrete_y(y, max_unique=20):
    y = np.asarray(y)
    return np.issubdtype(y.dtype, np.integer) or np.unique(y).size <= max_unique

def r2_rmse_cv(Z, Y, cv_splits=5, seed=42, alphas=(0.1,1.0,10.0)):
    Z = np.asarray(Z); Y = np.asarray(Y).ravel()
    Zs = StandardScaler().fit_transform(Z)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    model = RidgeCV(alphas=np.array(alphas))
    y_pred = cross_val_predict(model, Zs, Y, cv=cv, n_jobs=1)
    return float(r2_score(Y, y_pred)), float(np.sqrt(mean_squared_error(Y, y_pred)))

def mi_knn_continuous(Z, Y, k=5, random_state=42):
    Z = np.asarray(Z, float)
    y = np.asarray(Y, float).reshape(-1, 1)
    X = np.concatenate([Z, y], axis=1)
    N = X.shape[0]
    Xs = StandardScaler().fit_transform(X)
    Zs = StandardScaler().fit_transform(Z)
    ys = StandardScaler().fit_transform(y)
    if _HAS_SCIPY:
        dists, _ = cKDTree(Xs).query(Xs, k=k+1, p=np.inf)
        eps = dists[:, k] - 1e-15
        tree_z, tree_y = cKDTree(Zs), cKDTree(ys)
        n_z = np.array([len(tree_z.query_ball_point(Zs[i], r=eps[i], p=np.inf)) - 1 for i in range(N)], float)
        n_y = np.array([len(tree_y.query_ball_point(ys[i], r=eps[i], p=np.inf)) - 1 for i in range(N)], float)
    else:
        nn_joint = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1).fit(Xs)
        dists, _ = nn_joint.kneighbors(Xs)
        eps = dists[:, k] - 1e-15
        nn_z = NearestNeighbors(metric="chebyshev").fit(Zs)
        nn_y = NearestNeighbors(metric="chebyshev").fit(ys)
        n_z, n_y = np.empty(N), np.empty(N)
        for i in range(N):
            n_z[i] = len(nn_z.radius_neighbors(Zs[i:i+1], radius=eps[i], return_distance=False)[0]) - 1
            n_y[i] = len(nn_y.radius_neighbors(ys[i:i+1], radius=eps[i], return_distance=False)[0]) - 1
    return float(max(digamma(k) + digamma(N) - np.mean(digamma(n_z + 1) + digamma(n_y + 1)), 0.0))

def mi_discrete_lower_bound(Z, y, cv_splits=5, random_state=42, C=1.0):
    Z = np.asarray(Z, float); y = np.asarray(y).ravel()
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))  # nats
    Zs = StandardScaler().fit_transform(Z)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, n_jobs=1, multi_class="auto",
                             random_state=random_state)
    proba = cross_val_predict(clf, Zs, y, cv=cv, method="predict_proba", n_jobs=1)
    ce = log_loss(y, proba, labels=vals)
    return float(max(H - ce, 0.0)), float(H)

def outcome_sufficiency_internal_mi(Z, Y, cv_splits=5, seed=42, alphas=(0.1,1.0,10.0), mi_k=5):
    """Return R2, RMSE, and absolute MI (MI used internally for normalization only)."""
    Z = np.asarray(Z); Y = np.asarray(Y).ravel()
    R2, RMSE = r2_rmse_cv(Z, Y, cv_splits=cv_splits, seed=seed, alphas=alphas)
    if _is_discrete_y(Y):
        MI, H = mi_discrete_lower_bound(Z, Y, cv_splits=cv_splits, random_state=seed)
        return {"R2": R2, "RMSE": RMSE, "_MI_abs": float(MI), "_H_y": float(H)}
    else:
        MI = mi_knn_continuous(Z, Y, k=mi_k, random_state=seed)
        return {"R2": R2, "RMSE": RMSE, "_MI_abs": float(MI)}

def _entropy_discrete(y):
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())



def aggregate_over_files(FILES, *, input_features, lat_dim, hid_enc, scaler_x, dummies, save_dir):
    all_rows = []
    pooled_X, pooled_T, pooled_Y = [], [], []
    Z_components_pool = {"Ours": [], "DRCFR": [], "VAE": [], "CFR": []}

    for file_no in FILES:
        rows, pool = process_one_file(file_no,
                                      input_features=input_features,
                                      lat_dim=lat_dim,
                                      hid_enc=hid_enc,
                                      scaler_x=scaler_x,
                                      dummies=dummies)
        for r in rows:
            r["file_no"] = file_no
            all_rows.append(r)
        pooled_X.append(pool["X"]); pooled_T.append(pool["T"]); pooled_Y.append(pool["Y"])
        for m in Z_components_pool.keys():
            Z_components_pool[m].append(pool["Z_components"][m])

    X_pool = np.vstack(pooled_X)
    T_pool = np.concatenate(pooled_T)
    Y_pool = np.concatenate(pooled_Y)

    df = pd.DataFrame(all_rows)
    spaces = df["space"].unique().tolist()
    mean_std_rows = []
    for sp in spaces:
        sp_rows = df[df["space"] == sp].to_dict(orient="records")
        mean_std_rows.append(_agg_mean_std(sp_rows))
    agg_table = pd.DataFrame(mean_std_rows)
    agg_csv = os.path.join(save_dir, "metrics_agg_mean_std.csv")
    agg_table.to_csv(agg_csv, index=False)
    print(f"[Saved] {agg_csv}")

    return agg_table, X_pool, T_pool, Y_pool, Z_components_pool
#load the model
lat_dim=60
hid_enc=60
dummies=5
input_features=25+dummies
dim_range=lat_dim//4 #5,10,15
lat_dim=dim_range*4
fstart=0
fend=fstart+dim_range
sstart=fend
send=sstart+dim_range
tstart=send
tend=tstart+dim_range
frstart=tend
frend=frstart+dim_range
# ====================================
# MAIN
# ====================================
if __name__ == "__main__":
    set_global_seed(42)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Provide these from your codebase:
    #   - input_features, lat_dim, hid_enc, scaler_x, dummies
    #   - TarNet, TarNet_DRCFR, TarNet_VAE classes
    #   - get_data(split, file_no), add_dummy_features_shuffle(df, dummies)
    #   - load_masks(file_no) -> dict with mask_gamma, mask_delta, mask_upsilon
    #   - load_rep(file_no)   -> dict with 'rep' (CFR tensor)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Example config (edit as needed)
    FILES = list(range(1, 6))
    SAVE_DIR = f"PR_ana/DRI/ideal_01exp_IHDP_{dummies}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        scaler_x
    except NameError:
        scaler_x = StandardScaler()

    # Aggregate across files
    agg_table, X_pool, T_pool, Y_pool, Z_components_pool = aggregate_over_files(
        FILES,
        input_features=input_features,
        lat_dim=lat_dim,
        hid_enc=hid_enc,
        scaler_x=scaler_x,
        dummies=dummies,
        save_dir=SAVE_DIR
    )

    print("\n=== Mean ± Std across files ===")
    print(agg_table.to_string(index=False))

    # ===== Save separate panels for X and all methods =====
    out_single_dir = os.path.join(SAVE_DIR, "single_panels")
    save_all_panels_per_space(
        X_pool, T_pool, Y_pool,
        space_name="X", view_name="orig",
        out_dir=out_single_dir,
        k_mix=5, k_smooth=5, seed=42
    )

    for method_name, comps_list in Z_components_pool.items():
        merged = {}
        for comps in comps_list:
            for k, arr in comps.items():
                merged.setdefault(k, []).append(arr)
        merged = {k: np.vstack(v) for k, v in merged.items()}
        view_sel = VIEW_POLICY[method_name]["plot"]  # e.g., 'gamma+delta+upsilon' or 'all'
        Z_view = get_view(merged, view_sel)

        # separate figures
        save_all_panels_per_space(
            Z_view, T_pool, Y_pool,
            space_name=f"Z_{method_name}", view_name=view_sel.replace("+", "_"),
            out_dir=out_single_dir,
            k_mix=5, k_smooth=5, seed=42
        )

        # ===== NEW: combined single panel for the two methods (trust/cont + density) =====
        out_qc_dir = os.path.join(SAVE_DIR, "qc_panels")
        save_qc_panel_for_method(
            X_pool, Z_view, method_name, out_qc_dir,
            k_tc=5, k_density=5, seed=42
        )


# In[10]:


qc_plot_all_methods(
    X_pool, Z_components_pool,
    out_dir=os.path.join(SAVE_DIR, "qc"),
    view_key="plot",      # uses VIEW_POLICY[method]["plot"]
    k_tc=5,              # k for Trustworthiness/Continuity
    k_density=5,         # k for neighborhood density
    seed=42
)


# In[11]:


DASH_DIR = os.path.join(SAVE_DIR, f"dashboards_{dummies}")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DASH_DIR, exist_ok=True)
for method_name, comps_list in Z_components_pool.items():
    merged = {}
    for comps in comps_list:
        for k, arr in comps.items():
            merged.setdefault(k, []).append(arr)
    merged = {k: np.vstack(v) for k, v in merged.items()}
    view_sel = VIEW_POLICY[method_name]["plot"]  # e.g., 'gamma+delta+upsilon' or 'all'
    Z_view = get_view(merged, view_sel)

    fig = make_embedding_dashboard(
    Z_view, T_pool, Y_pool,
    method_name=method_name, view_name=view_sel,
    random_state=42, k_mix=10, figsize=(12, 10), bins_mix=20, vmax_cov=None
    )
    out_path = os.path.join(DASH_DIR, f"dashboard_{method_name}_{view_sel}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")








