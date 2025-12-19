# Auto-generated from Diagnostic_demo_Clear26.ipynb
# Notebook cells have been linearized into a plain Python script.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
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
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


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


class Regressors(nn.Module):
    def __init__(self,
                 input_dim,hid_dim,
                 regularization):
        super(Regressors, self).__init__()



        self.regressor1_y0 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressor2_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.regressor3_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressor4_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressor5_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )


        self.regressor6_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressorO_y0 = nn.Linear(hid_dim, 1)



        self.regressor1_y1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.regressor2_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressor3_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressor4_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressor5_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )

        self.regressor6_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),

            nn.ELU(),
            nn.Dropout(p=regularization),
        )


        self.regressorO_y1 = nn.Linear(hid_dim, 1)




    def forward(self, inputs):

        # Regressors
        #del_ups=torch.cat((phi_delta, phi_upsilon), 1)
        out_y0 = self.regressor1_y0(inputs)
        out_y0 = self.regressor2_y0(out_y0)
        out_y0 = self.regressor3_y0(out_y0)
        #out_y0 = self.regressor4_y0(out_y0)  
        #out_y0 = self.regressor5_y0(out_y0)
        #out_y0 = self.regressor6_y0(out_y0)
        y0 = self.regressorO_y0(out_y0)

        out_y1 = self.regressor1_y1(inputs)
        out_y1 = self.regressor2_y1(out_y1)
        out_y1 = self.regressor3_y1(out_y1)
        #out_y1 = self.regressor4_y1(out_y1)
        #out_y1 = self.regressor5_y1(out_y1)
        #out_y1 = self.regressor6_y1(out_y1)

        y1 = self.regressorO_y1(out_y1)

        # classifires
        #gam_del=torch.cat((phi_gamma,phi_delta), 1)
        #out_w=self.classifier_w1(phi_delta)
        #out_w=self.classifier_w2(out_w)
        #out_w_f=self.sig(self.classifier_w3(out_w))

        #out_t=self.classifier_t1(gam_del)
        #out_t=self.classifier_t2(out_t)
        #out_t_f=self.sig(self.classifier_t3(out_t))

        # Returning arguments

        concat = torch.cat((y0, y1), 1)
        return concat#out_w_f,out_t_f


# In[7]:


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


# In[8]:


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


# In[9]:


# -*- coding: utf-8 -*-
"""
ΔŶ contour (treatment-effect surface) using ONLY provided tensors/arrays:
Inputs you ALREADY have:
    Z   : (N, d) latent codes (tensor or numpy)
    y0  : (N,)   predicted outcomes at t=0
    y1  : (N,)   predicted outcomes at t=1

What it does:
  • PCA→2D for visualization
  • Fit tiny surrogates (Z -> y0) and (Z -> y1)  [ridge or kNN]
  • Predict on a dense latent grid and plot ΔŶ = y1 - y0 as a contour surface
  • Overlay points colored by their own ΔŶ
Saves: te_delta_contours.svg/.png

Dependencies: numpy, matplotlib, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

# ---------------- small utils ----------------
def _np(x):
    try:
        import torch
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _make_pca_grid(Z, res=220, pct=(1, 99)):
    """Fit PCA(2) on Z, build a 2D grid in PCA plane, back-map to latent space."""
    Z = _np(Z).astype(float)
    pca = PCA(n_components=2).fit(Z - Z.mean(0, keepdims=True))
    V2  = pca.components_.T
    Z2  = (Z - Z.mean(0)) @ V2
    x0, x1 = np.percentile(Z2[:, 0], pct)
    y0, y1 = np.percentile(Z2[:, 1], pct)
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    XX, YY = np.meshgrid(xs, ys)
    grid2  = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Zg     = grid2 @ V2.T + Z.mean(0, keepdims=True)   # (res^2, d)
    return Z2, grid2, Zg

def _plot_contour(grid2, surface, Z2, dot_colors, title, cbar_label="ΔŶ", fname="te_delta_contours"):
    """Filled contour + scatter overlay; saves SVG/PNG."""
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    cf = ax.tricontourf(grid2[:, 0], grid2[:, 1], surface, levels=18, cmap="coolwarm")
    plt.colorbar(cf, ax=ax, label=cbar_label)
    ax.scatter(Z2[:, 0], Z2[:, 1], c=dot_colors, s=12, cmap="coolwarm",
               edgecolor="k", linewidth=0.15, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    #plt.savefig(f"{fname}.svg", bbox_inches="tight", dpi=300)
    plt.savefig(f"{fname}.png", bbox_inches="tight", dpi=500)
    plt.close(fig)

# ---------------- main function ----------------
def plot_deltaY_from_latent(
    Z,             # (N, d) latent codes you already have
    y0,            # (N,)   predicted Ŷ at t=0 you already have
    y1,            # (N,)   predicted Ŷ at t=1 you already have
    surrogate="ridge",   # "ridge" or "knn"
    grid_res=220,
    out_prefix="te"
):
    """
    Builds ΔŶ surface in latent PCA plane using only (Z, y0, y1).
    Returns dict with intermediate arrays.
    """
    Z = _np(Z); y0 = _np(y0).reshape(-1); y1 = _np(y1).reshape(-1)
    assert Z.shape[0] == y0.shape[0] == y1.shape[0], "Z, y0, y1 must align."

    # PCA plane + grid
    Z2, grid2, Zg = _make_pca_grid(Z, res=grid_res)

    # Fit tiny surrogates Z -> y0 and Z -> y1
    if surrogate == "ridge":
        regr0 = Ridge(alpha=1.0)
        regr1 = Ridge(alpha=1.0)
    elif surrogate == "knn":
        regr0 = KNeighborsRegressor(n_neighbors=50, weights="distance")
        regr1 = KNeighborsRegressor(n_neighbors=50, weights="distance")
    else:
        raise ValueError("surrogate must be 'ridge' or 'knn'")

    regr0.fit(Z, y0)
    regr1.fit(Z, y1)

    # Grid predictions and ΔŶ surface
    y0_grid = regr0.predict(Zg)
    y1_grid = regr1.predict(Zg)
    delta_grid = y1_grid - y0_grid

    # Dot colors = per-point ΔŶ
    delta_pts = y1 - y0

    # Plot & save
    tag = f"{out_prefix}_delta_effect_contours"
    _plot_contour(
        grid2, delta_grid, Z2, delta_pts,
        title="Treatment-effect surface  ΔŶ = Ŷ(t=1) − Ŷ(t=0)",
        cbar_label="ΔŶ",
        fname=tag
    )

    return {
        "Z": Z, "Z2": Z2,
        "grid2": grid2, "Z_grid": Zg,
        "y0_grid": y0_grid, "y1_grid": y1_grid, "delta_grid": delta_grid,
        "delta_pts": delta_pts
    }

# ---------------- optional: single-outcome surfaces from (Z, y_t) ----------------
def plot_outcome_from_latent(
    Z, y_t, surrogate="ridge", grid_res=220, out_prefix="diagnostic/cc", tag="t0"
):
    """
    Makes a single outcome surface (e.g., t=0 or t=1) using only (Z, y_t).
    Saves: {out_prefix}_outcome_contours_{tag}.svg/.png
    """
    Z = _np(Z); y_t = _np(y_t).reshape(-1)
    Z2, grid2, Zg = _make_pca_grid(Z, res=grid_res)

    if surrogate == "ridge":
        regr = Ridge(alpha=1.0)
    elif surrogate == "knn":
        regr = KNeighborsRegressor(n_neighbors=50, weights="distance")
    else:
        raise ValueError("surrogate must be 'ridge' or 'knn'")

    regr.fit(Z, y_t)
    y_grid = regr.predict(Zg)

    _plot_contour(
        grid2, y_grid, Z2, y_t,
        title=f"Outcome surface (tag={tag}) in PCA plane",
        cbar_label="Ŷ",
        fname=f"{out_prefix}_outcome_contours_{tag}"
    )

def prepare_data_for_file(file_no, dummies):
    x_data_tr, y_data_tr = get_data('train', file_no)
    x_data_n = add_dummy_features_shuffle(x_data_tr, dummies)

    # normalize only covariates (from col 5 onward), keep first 5 as is
    x_data_n_nor = pd.DataFrame(scaler_x.fit_transform(x_data_n.iloc[:, 5:]))
    data_train_tran = pd.concat([x_data_n.iloc[:, 0:5], x_data_n_nor], axis=1)

    data_np = data_train_tran.to_numpy()
    data_t = torch.from_numpy(data_np.astype(np.float32)).to(device)

    X_t  = data_t[:, 5:]   # covariates
    X_np = data_np[:, 5:]
    T_np = data_np[:, 0]
    Y_np = data_np[:, 1]

    return X_t, X_np, T_np, Y_np  

def load_masks(file_no, out_dir="saved_masks", device="cuda:0"):
    path = f"{out_dir}/IHDP_masks_{file_no}_{dummies}.pt"
    data = torch.load(path, map_location=device)
    return {k: v.to(device) for k, v in data.items()}

def ensure_dir_exists(path_prefix):
    """Ensure the directory for given prefix (e.g., diagnostic/contour_xyz) exists."""
    dir_path = os.path.dirname(path_prefix)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def _np(x):
    try:
        import torch
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _cosine(a, b, eps=1e-8):
    a = _np(a).astype(float); b = _np(b).astype(float)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def counterfactual_consistency_arrays(Z0, Z1, y0, y1):
    Z0 = _np(Z0); Z1 = _np(Z1); y0 = _np(y0).reshape(-1); y1 = _np(y1).reshape(-1)
    assert Z0.shape == Z1.shape and Z0.shape[0] == y0.shape[0] == y1.shape[0], "Shape mismatch."
    dZ = Z1 - Z0
    norms = np.linalg.norm(dZ, axis=1)
    cosines = np.array([_cosine(Z0[i], Z1[i]) for i in range(Z0.shape[0])], dtype=float)
    delta_y = y1 - y0
    return norms, cosines, delta_y


"""
# ---------- plotting ----------
def plot_cc_histograms(Z0, Z1, y0, y1, out_prefix="cc", bins=40):
    norms, cosines, delta_y = counterfactual_consistency_arrays(Z0, Z1, y0, y1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    # ‖ΔZ‖
    ax = axes[0]
    ax.hist(norms, bins=bins, alpha=0.9)
    ax.set_title(r"$\| \Delta Z \|$ (latent displacement)")
    ax.set_xlabel(r"$\| z(t{=}1) - z(t{=}0) \|$")
    ax.set_ylabel("Count")
    ax.axvline(norms.mean(), linestyle="--", linewidth=1.2)
    ax.text(0.98, 0.95, f"mean={norms.mean():.3f}\nstd={norms.std():.3f}",
            transform=ax.transAxes, ha="right", va="top")

    # cosine
    ax = axes[1]
    ax.hist(cosines, bins=bins, alpha=0.9)
    ax.set_title(r"cosine$(z_0,z_1)$ (geometric smoothness)")
    ax.set_xlabel("cosine similarity")
    ax.axvline(cosines.mean(), linestyle="--", linewidth=1.2)
    ax.text(0.98, 0.95, f"mean={cosines.mean():.3f}\nstd={cosines.std():.3f}",
            transform=ax.transAxes, ha="right", va="top")
    ax.set_xlim(-1.0, 1.0)

    # ΔŶ
    ax = axes[2]
    ax.hist(delta_y, bins=bins, alpha=0.9)
    ax.set_title(r"$\Delta \hat{Y} = \hat{Y}(1)-\hat{Y}(0)$ (ITE)")
    ax.set_xlabel(r"$\Delta \hat{Y}$")
    ax.axvline(delta_y.mean(), linestyle="--", linewidth=1.2)
    ax.text(0.98, 0.95, f"mean={delta_y.mean():.3f}\nstd={delta_y.std():.3f}",
            transform=ax.transAxes, ha="right", va="top")

    plt.tight_layout()
    #plt.savefig(f"{out_prefix}_histograms.svg", bbox_inches="tight", dpi=300)
    plt.savefig(f"{out_prefix}_histograms.png", bbox_inches="tight", dpi=500)
    plt.close(fig)
"""
def plot_cc_histograms(Z0, Z1, y0, y1, out_prefix="cc", bins=40):
    norms, cosines, delta_y = counterfactual_consistency_arrays(Z0, Z1, y0, y1)

    # ---------------------------
    # 1. ‖ΔZ‖ histogram
    # ---------------------------
    plt.figure(figsize=(5,4))
    plt.hist(norms, bins=bins, alpha=0.9)
    plt.title(r"$\| \Delta Z \|$ (latent displacement)")
    plt.xlabel(r"$\| z(t{=}1) - z(t{=}0) \|$")
    plt.ylabel("Count")
    plt.axvline(norms.mean(), linestyle="--", linewidth=1.2)
    plt.text(0.98, 0.95, f"mean={norms.mean():.3f}\nstd={norms.std():.3f}",
             transform=plt.gca().transAxes, ha="right", va="top")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_norms.png", bbox_inches="tight", dpi=500)
    plt.close()

    # ---------------------------
    # 2. cosine similarity histogram
    # ---------------------------
    plt.figure(figsize=(5,4))
    plt.hist(cosines, bins=bins, alpha=0.9)
    plt.title(r"cosine$(z_0,z_1)$ (geometric smoothness)")
    plt.xlabel("cosine similarity")
    plt.axvline(cosines.mean(), linestyle="--", linewidth=1.2)
    plt.text(0.98, 0.95, f"mean={cosines.mean():.3f}\nstd={cosines.std():.3f}",
             transform=plt.gca().transAxes, ha="right", va="top")
    plt.xlim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cosine.png", bbox_inches="tight", dpi=500)
    plt.close()

    # ---------------------------
    # 3. ΔŶ histogram
    # ---------------------------
    plt.figure(figsize=(5,4))
    plt.hist(delta_y, bins=bins, alpha=0.9)
    plt.title(r"$\Delta \hat{Y} = \hat{Y}(1)-\hat{Y}(0)$ (ITE)")
    plt.xlabel(r"$\Delta \hat{Y}$")
    plt.axvline(delta_y.mean(), linestyle="--", linewidth=1.2)
    plt.text(0.98, 0.95, f"mean={delta_y.mean():.3f}\nstd={delta_y.std():.3f}",
             transform=plt.gca().transAxes, ha="right", va="top")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_deltaY.png", bbox_inches="tight", dpi=500)
    plt.close()

def plot_cc_joint(Z0, Z1, y0, y1, out_prefix="cc"):
    norms, _, delta_y = counterfactual_consistency_arrays(Z0, Z1, y0, y1)
    # Pearson r
    r = np.corrcoef(norms, delta_y)[0, 1]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(norms, delta_y, s=12, alpha=0.75, edgecolor="k", linewidth=0.15)
    ax.set_title(r"Joint: $\| \Delta Z \|$ vs. $\Delta \hat{Y}$")
    ax.set_xlabel(r"$\| \Delta Z \|$")
    ax.set_ylabel(r"$\Delta \hat{Y}$")
    ax.text(0.98, 0.05, f"Pearson r = {r:.3f}", transform=ax.transAxes, ha="right", va="bottom")
    plt.tight_layout()
    #plt.savefig(f"{out_prefix}_joint_scatter.svg", bbox_inches="tight", dpi=300)
    plt.savefig(f"{out_prefix}_joint_scatter.png", bbox_inches="tight", dpi=500)

def make_cc_pairs_from_matching(Z, T_np, y0_hat, y1_hat):
    """
    Build pseudo (Z0, Z1, y0, y1) pairs from:
      - Z      : (N, d) latent tensor
      - T_np   : (N,) treatment numpy array (0/1)
      - y0_hat : (N,) potential outcome under t=0 (tensor)
      - y1_hat : (N,) potential outcome under t=1 (tensor)

    Strategy:
      - For each treated unit, find nearest control in Z-space.
      - Pair (control -> Z0, treated -> Z1) as a pseudo-counterfactual pair.
    """
    Z_np = _np(Z)                       # uses your helper
    T_np = np.asarray(T_np).reshape(-1)

    y0 = _np(y0_hat).reshape(-1)
    y1 = _np(y1_hat).reshape(-1)

    treated_idx = np.where(T_np == 1)[0]
    control_idx = np.where(T_np == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Need both treated and control units for matching-based CC.")

    # Fit NN on controls
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(Z_np[control_idx])

    # For each treated point, find nearest control in latent space
    dists, nn_idx = nbrs.kneighbors(Z_np[treated_idx])
    matched_control_idx = control_idx[nn_idx[:, 0]]

    # Build pair arrays
    Z1_pairs = Z_np[treated_idx]               # latent of treated units
    Z0_pairs = Z_np[matched_control_idx]       # latent of matched controls

    y1_pairs = y1[treated_idx]                 # Y(1) for treated
    y0_pairs = y0[matched_control_idx]         # Y(0) for matched controls

    return Z0_pairs, Z1_pairs, y0_pairs, y1_pairs

def compute_empirical_ate(y, t):
    """
    Empirical ATE = mean outcome difference between treated and control groups
    """
    y = np.asarray(y)
    t = np.asarray(t)
    ate_empirical = np.mean(y[t == 1]) - np.mean(y[t == 0])
    return ate_empirical
def load_rep(file_no, out_dir="introduction_to_CFR-main/saved_reps/", device="cuda:0"):
    path = f"{out_dir}/IHDP_rep_{file_no}_{dummies}.pt"
    data = torch.load(path, map_location=device)
    return {k: v.to(device) for k, v in data.items()}
def load_y(file_no, out_dir="introduction_to_CFR-main/saved_reps/", device="cuda:0"):
    path = f"{out_dir}/IHDP_y_{file_no}_{dummies}.pt"
    data = torch.load(path, map_location=device)
    return {k: v.to(device) for k, v in data.items()}


# In[10]:


model_configs = {
     "CFR": {
        "enc_cls": False,          # <-- use your actual encoder class
        "reg_cls": False,      # <-- use your actual regressor class
        "enc_ckpt": "PR_ana/CFR_IHDP_{file_no}_{dummies}.pth",
        "reg_ckpt": "PR_ana/reg_CFR_IHDP_{file_no}_{dummies}.pth",
    },

    "DRCFR": {
        "enc_cls": TarNet_DRCFR,
        "reg_cls": Regressors,
        "enc_ckpt": "PR_ana/DRCFR_IHDP_{file_no}_{dummies}.pth",
        "reg_ckpt": "PR_ana/reg_DRCFR_IHDP_{file_no}_{dummies}.pth",
    },
    "TEDVAE": {
        "enc_cls": TarNet_VAE,      # <-- your TEDVAE encoder class
        "reg_cls": Regressors,   # <-- or whatever you used
        "enc_ckpt": "PR_ana/VAE_IHDP_{file_no}_{dummies}.pth",
        "reg_ckpt": "PR_ana/reg_VAE_IHDP_{file_no}_{dummies}.pth",
    },
    "GLOVE-ITE": {
        "enc_cls": TarNet_DRI,    # <-- your GLOVE-ITE encoder class
        "reg_cls": Regressors, # <-- your GLOVE-ITE head / regressor
        "enc_ckpt": "PR_ana/DRI_ITE_IHDP_{file_no}_{dummies}.pth",
        "reg_ckpt": "PR_ana/reg_DRI_ITE_IHDP{file_no}_{dummies}.pth",
    },
}

def _to_np(x):
    """Convert tensor or array/list to np.ndarray."""
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------
# 3. Main loop: over models and files (1..5)
# ---------------------------------------------------
if __name__ == "__main__":
    lat_dim = 60
    hid_enc = 60
    dummies = 5
    hid_y = 100
    y_input = lat_dim * 2
    input_features = 25 + dummies
    dim_range = lat_dim // 4  # 5,10,15
    lat_dim = dim_range * 4
    fstart = 0
    fend = fstart + dim_range
    sstart = fend
    send = sstart + dim_range
    tstart = send
    tend = tstart + dim_range
    frstart = tend
    frend = frstart + dim_range

    file_nos = [1, 2, 3, 4, 5]   # or whatever your 5 files are
    ATE=[]

    for model_name, cfg in model_configs.items():
        print(f"\n=== Processing model: {model_name} ===")

        # to pool over 5 files (for "average" plots)
        all_Z   = []
        all_y0  = []
        all_y1  = []
        all_T   = []   # need T for cc plots

        CFR = False

        for file_no in file_nos:
            print(f"  -> File {file_no}")

            # ----- load model for this file -----
            if model_name != 'CFR':
                if model_name != 'GLOVE-ITE':
                    Enc = cfg["enc_cls"](input_features, lat_dim, 0.1).to(device)
                    Enc.load_state_dict(torch.load(
                        cfg["enc_ckpt"].format(file_no=file_no, dummies=dummies),
                        map_location=device
                    ))
                    Enc.eval()
                else:
                    Enc = cfg["enc_cls"](input_features, hid_enc, 0.1).to(device)
                    Enc.load_state_dict(torch.load(
                        cfg["enc_ckpt"].format(file_no=file_no, dummies=dummies),
                        map_location=device
                    ))
                    Enc.eval()

                Reg = cfg["reg_cls"](y_input, hid_y, 0.1).to(device)
                Reg.load_state_dict(torch.load(
                    cfg["reg_ckpt"].format(file_no=file_no, dummies=dummies),
                    map_location=device
                ))
                Reg.eval()

            # ----- prepare data -----
            X_t, X_np, T_np, Y_np = prepare_data_for_file(file_no, dummies)
            print('ATE:', compute_empirical_ate(Y_np, T_np))
            if(model_name == 'CFR'):
                ATE.append(compute_empirical_ate(Y_np, T_np))

            # ----- get latent and predictions -----
            with torch.no_grad():
                print(model_name)

                if model_name == 'TEDVAE':
                    (gV, dV, uV, muV, lvV,
                     phi_delta_mean, phi_delta_var,
                     phi_upsilon_mean, phi_upsilon_var) = Enc(X_t)
                    Z = torch.cat((dV, uV), dim=1)

                elif model_name == 'GLOVE-ITE':
                    go, do, uo, oo = Enc(X_t)
                    Z = torch.cat((do, uo), dim=1)

                elif model_name == 'CFR':
                    repre = load_rep(file_no)
                    Z = repre["rep"]      # assume np array or tensor
                    CFR = True

                else:
                    gD, dD, uD = Enc(X_t)
                    Z = torch.cat((dD, uD), dim=1)

                # outcomes
                if not CFR:
                    y_pred = Reg(Z)      # shape (N, 2): [y0_hat, y1_hat]
                    y0_hat = y_pred[:, 0]
                    y1_hat = y_pred[:, 1]
                else:
                    y = load_y(file_no)
                    y0_hat = y['y0']     # could be numpy
                    y1_hat = y['y1']

            # -----------------------------
            # (A) Per-file plot (optional)
            # -----------------------------
            out_prefix_file = f"diagnostic_2/DRI/{model_name}/contour_{model_name}_file{file_no}"
            ensure_dir_exists(out_prefix_file)
            plot_deltaY_from_latent(
                Z,
                y0_hat,
                y1_hat,
                surrogate="ridge",
                out_prefix=out_prefix_file
            )

            cc_prefix_file = f"diagnostic_2/DRI/{model_name}/cc_file{file_no}"
            ensure_dir_exists(cc_prefix_file)

            Z0_pairs, Z1_pairs, y0_pairs, y1_pairs = make_cc_pairs_from_matching(
                Z, T_np, y0_hat, y1_hat
            )

            plot_cc_histograms(
                Z0_pairs, Z1_pairs,
                y0_pairs, y1_pairs,
                out_prefix=cc_prefix_file
            )

            plot_cc_joint(
                Z0_pairs, Z1_pairs,
                y0_pairs, y1_pairs,
                out_prefix=cc_prefix_file
            )

            # -----------------------------
            # (B) Accumulate for "avg over 5 files"
            # -----------------------------
            all_Z.append(_to_np(Z))
            all_y0.append(_to_np(y0_hat))
            all_y1.append(_to_np(y1_hat))
            all_T.append(np.asarray(T_np))

        # --------------------------------------
        # 4. “Average over 5 files” plots
        #    (pooled across all units)
        # --------------------------------------
        Z_pool  = np.concatenate(all_Z, axis=0)
        y0_pool = np.concatenate(all_y0, axis=0)
        y1_pool = np.concatenate(all_y1, axis=0)
        T_pool  = np.concatenate(all_T, axis=0)

        avg_prefix = f"diagnostic_2/DRI/{model_name}/avg5_{model_name}"
        ensure_dir_exists(avg_prefix)

        # (1) ΔŶ surface: one figure per method
        plot_deltaY_from_latent(
            Z_pool,
            y0_pool,
            y1_pool,
            surrogate="ridge",
            out_prefix=f"{avg_prefix}_contour"
        )

        # (2) Counterfactual-consistency pairs on pooled data
        Z0_pool, Z1_pool, y0_pool_pairs, y1_pool_pairs = make_cc_pairs_from_matching(
            Z_pool, T_pool, y0_pool, y1_pool
        )

        # (3) One histogram figure per method
        plot_cc_histograms(
            Z0_pool, Z1_pool,
            y0_pool_pairs, y1_pool_pairs,
            out_prefix=f"{avg_prefix}_cc"
        )

        # (4) One joint plot figure per method
        plot_cc_joint(
            Z0_pool, Z1_pool,
            y0_pool_pairs, y1_pool_pairs,
            out_prefix=f"{avg_prefix}_cc"
        )


# In[11]:


print(np.mean(ATE),np.std(ATE))


# In[ ]:




