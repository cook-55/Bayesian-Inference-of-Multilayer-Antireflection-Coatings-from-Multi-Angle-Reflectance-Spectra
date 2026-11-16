"""
光学薄膜反演方法对比实验：ASMC vs Gaussian VA
投稿版可视化与数值记录 (Joint-KDE + Zoom-in Insets)

本版本按 Applied Optics 两位审稿人意见完成的修订版：
- R2-1 / R2-minor-1 : 修复 TMM 反射率公式（入射介质导纳 + 斜入射基底导纳 +
  显式偏振处理），通过裸基底菲涅尔极限自检；
- R2-2 / R2-minor-2 : 删除 HfO2 / Ta2O5 的 Sellmeier 钳位与 ULD 表层
  n_surface 钳位（原钳位使色散与孔隙率信息失效）；
- R1-1 / R2-4       : 删除 632.8 nm 十倍波长加权，两法统一为同一标准
  高斯似然（两角度平方和，含 1/sigma^2）；
- R2-5 / R2-8       : Gaussian VA 改为逐样本 Monte-Carlo ELBO（数据项 +
  对均匀盒先验的变换分布 KL），后验宽度随噪声水平变化；
- R2-6              : ASMC 移动核加 Metropolis 接受步（对退火目标不变），
  终估按最终权重重采样；线性退火改为按 ESS 目标的自适应退火；
- R1-2 / R1-5       : 报告全部 11 个参数（含折射率漂移）、两角度光谱
  RMSE、RMSE/sigma 与 2-sigma 覆盖率，主流程支持多种子统计；
- R1-4              : 增加全协方差 Gaussian VA 变体与后验相关性分析；
- R2-8              : 增加独立随机游走 Metropolis 参照解；
- R1-3 / R1-6 / R2-7: 提供 ASMC 超参、VA 超参、起始值、孔隙率前向
  敏感性分析工具；
- 可复现性          : 噪声改在 CPU 端由固定种子流生成（跨设备一致），
  运行时间统计加入 CUDA 同步。
"""

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.transforms import SigmoidTransform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns
from scipy.stats import gaussian_kde, truncnorm
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

warnings.filterwarnings('ignore')
OUTPUT_DIR = Path(__file__).resolve().parent / "figs"

# 11 维参数向量标签（d1..d5, dn1..dn5, f），用于全参数报告与相关性分析
PARAM_LABELS = ['d1_ULD', 'd2_HfO2', 'd3_SiO2', 'd4_Ta2O5', 'd5_MgF2',
                'dn1', 'dn2', 'dn3', 'dn4', 'dn5', 'f']


# ============================================================
# 全局配置
# ============================================================

@dataclass
class Config:
    """实验配置"""
    # 物理参数
    DESIGN_WAVELENGTH: float = 632.8
    WAVELENGTH_RANGE: Tuple[float, float] = (550, 720)
    SUBSTRATE_N: float = 1.515
    AIR_N: float = 1.0

    # 膜系结构（5层）
    N_LAYERS: int = 5
    MATERIALS: List[str] = None

    # 蛾眼参数
    MOTH_EYE_GRAD_STRENGTH: float = 0.965
    TRUE_F_ULD: float = 0.345

    # 噪声水平（核心可配置项）
    SIGMA_OBS: float = 1e-5

    # 斜入射通道偏振 (R2-minor-1)：'s' = TE（原实现约定），'p' = TM，
    # 'unpolarized' = 两偏振反射率平均
    POLARIZATION: str = 's'

    # 计算设备
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ASMC参数
    N_WAVELENGTHS: int = 120
    N_PARTICLES_SMC: int = 4000
    T_STAGES_SMC: int = 300            # 自适应退火的最大退火级数上限（保险丝）
    ESS_THRESHOLD_FRAC: float = 0.5    # 自适应退火每级的目标 ESS (x N)，每级重采样
    JITTER_SCALE: float = 0.5          # 坐标级 MH 提议尺度（相对各维边缘标准差）
    MH_SWEEPS: int = 2                 # 每级坐标 MH 扫描次数
    FINAL_SWEEPS: int = 3              # beta=1 后的精修扫描次数

    # Gaussian VA hyperparameters
    EPOCHS_VA: int = 5000
    LR_VA: float = 5e-4
    N_SAMPLES_VA: int = 16
    KL_WEIGHT: float = 1.0            # 均匀盒先验的 KL 权重
    LR_STD_MULTIPLIER: float = 50.0   # log_std 参数的学习率倍率（见 GaussianVA.run）

    # 潜变量 log_std 的数值下界。
    STD_LOG_MIN: float = -18.0

    # 先验可行域（R2-3：两法共用的单一先验定义）
    BOUNDS_D: Tuple[float, float] = (40.0, 150.0)    # 厚度 (nm)
    BOUNDS_DN: Tuple[float, float] = (-0.02, 0.02)   # 折射率漂移
    BOUNDS_F: Tuple[float, float] = (0.30, 0.50)     # 孔隙率

    # 初始提议和变分中心的物理尺度
    INIT_SIGMA_D: float = 0.5         # nm
    INIT_SIGMA_DN: float = 0.002
    INIT_SIGMA_F: float = 0.005
    INIT_D_SCALE: float = 1.0         # 起始厚度相对设计值的缩放（起始值敏感性用）
    INIT_F_VALUE: float = None        # 起始孔隙率（None = TRUE_F_ULD）

    # 独立参照 MCMC (R2-8)
    REF_MCMC_STEPS: int = 20000
    REF_MCMC_BURNIN: int = 5000

    def __post_init__(self):
        if self.MATERIALS is None:
            self.MATERIALS = ['SiO2-ULD', 'HfO2', 'SiO2', 'Ta2O5', 'MgF2']

    def parameter_box(self):
        return np.array([self.BOUNDS_D] * 5 + [self.BOUNDS_DN] * 5 + [self.BOUNDS_F]).T

    def starting_parameters(self):
        thickness = self.DESIGN_WAVELENGTH / (4 * np.array([1.45, 1.95, 1.45, 2.05, 1.38]))
        porosity = self.TRUE_F_ULD if self.INIT_F_VALUE is None else self.INIT_F_VALUE
        center = np.r_[thickness * self.INIT_D_SCALE, np.zeros(5), porosity]
        scale = np.array([self.INIT_SIGMA_D] * 5 + [self.INIT_SIGMA_DN] * 5 + [self.INIT_SIGMA_F])
        return center, scale


# ============================================================
# 物理模型
# ============================================================

class PhysicsModel:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

    def sellmeier_sio2(self, wl_nm):
        # R2-2: 仅保留数值保护下限（在 550-720 nm 内 n>=1.455，不激活），
        # 不再对物理取值做区间钳位
        wl = wl_nm * 1e-3
        B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
        C1, C2, C3 = 0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2
        n2 = 1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2) + B3 * wl ** 2 / (wl ** 2 - C3)
        return torch.sqrt(torch.clamp(n2, min=1.45 ** 2))

    def sellmeier_hfo2(self, wl_nm):
        # R2-2: 删除原 [1.94, 1.96] 钳位。原 Sellmeier 系数在 550-720 nm
        # 给出 n ~= 1.751，恒低于钳位下界，等于把色散替换为常数 1.94，
        # 物理上自相矛盾
        wl = wl_nm * 1e-3
        B1, B2, B3 = 1.50, 0.55, 1.20
        C1, C2, C3 = 0.05 ** 2, 0.10 ** 2, 8.00 ** 2
        n2 = 1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2) + B3 * wl ** 2 / (wl ** 2 - C3)
        return torch.sqrt(n2)

    def sellmeier_ta2o5(self, wl_nm):
        # R2-2: 删除原 [2.08, 2.12] 钳位（同上，原系数给出 n ~= 1.941）
        wl = wl_nm * 1e-3
        B1, B2, B3 = 2.13, 0.58, 1.54
        C1, C2, C3 = 0.062 ** 2, 0.16 ** 2, 13.73 ** 2
        n2 = 1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2) + B3 * wl ** 2 / (wl ** 2 - C3)
        return torch.sqrt(n2)

    def sellmeier_mgf2(self, wl_nm):
        # R2-2: [1.37, 1.38] 在波段内不激活（n ~= 1.377-1.379），保留作为
        # 数值保护
        wl = wl_nm * 1e-3
        B1, B2 = 0.48755108, 0.39875031
        C1, C2 = 0.04338408 ** 2, 0.09461442 ** 2
        n2 = 1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2)
        return torch.clamp(torch.sqrt(n2), min=1.37, max=1.38)

    def get_base_index(self, wl):
        n_sio2 = self.sellmeier_sio2(wl)
        n_hfo2 = self.sellmeier_hfo2(wl)
        n_ta2o5 = self.sellmeier_ta2o5(wl)
        n_mgf2 = self.sellmeier_mgf2(wl)
        return torch.stack([n_sio2, n_hfo2, n_sio2, n_ta2o5, n_mgf2], dim=-1)

    def moth_eye_effective_n(self, n_uld, f, grad_strength=None):
        """R2-minor-2: 多孔 ULD SiO2 层的有效折射率模型。

        Maxwell-Garnett 均匀球孔近似给出基底有效折射率 n_base，再经
        n_surface = 1.09 + (n_base - 1.09) * (1 - alpha^2.2) 映射到表层。
        该层在 TMM 中作为单一均匀层处理（折射率不随深度变化），蛾眼梯度
        仅由标量 alpha 概括。

        原实现对 n_surface 施加的 [1.09, 1.10] 钳位已删除：该钳位在
        f 属于先验 [0.30, 0.50] 的大半区间内饱和，使光谱对孔隙率几乎
        完全不敏感（R1-6 / R2-minor-2），是原稿孔隙率误差居高不下的
        主要模型原因。
        """
        if grad_strength is None:
            grad_strength = self.config.MOTH_EYE_GRAD_STRENGTH
        f_sio2 = 1 - f
        eps_m = self.config.AIR_N ** 2
        eps_i = n_uld ** 2
        eps_eff = eps_m * ((eps_i + 2 * eps_m + 2 * f_sio2 * (eps_i - eps_m)) /
                           (eps_i + 2 * eps_m - f_sio2 * (eps_i - eps_m)))
        n_base = torch.sqrt(torch.clamp(eps_eff.real, min=1e-6))
        n_surface = 1.09 + (n_base - 1.09) * (1 - grad_strength ** 2.2)
        return n_surface

    def tmm_reflectance(self, d, dn, f, wl, theta_deg):
        """R2-1 / R2-minor-1: 修正后的传输矩阵反射率算子。

        原实现的 r = (m11 + m12*ns - m21/ns - m22) / (m11 + m12*ns + ...) 把
        基底导纳放在了入射介质导纳的位置上：裸基底反射率恒为 0（应为
        4.19%），理想增透反而反射 4.19%。修正为教科书形式

            r = (eta_0 * B - C) / (eta_0 * B + C),
            B = m11 + m12 * eta_s,  C = m21 + m22 * eta_s,

        其中 eta_s 为基底导纳、eta_0 为入射介质导纳，斜入射时按偏振取
        s: eta = n cos(theta) 或 p: eta = n / cos(theta)。厚度全为零时
        严格退化为单界面菲涅尔反射率（validate_forward_model 自检）。
        """
        if self.config.POLARIZATION == 'unpolarized':
            r_s = self._tmm_reflectance_pol(d, dn, f, wl, theta_deg, 's')
            r_p = self._tmm_reflectance_pol(d, dn, f, wl, theta_deg, 'p')
            return 0.5 * (r_s + r_p)
        return self._tmm_reflectance_pol(d, dn, f, wl, theta_deg, self.config.POLARIZATION)

    def _tmm_reflectance_pol(self, d, dn, f, wl, theta_deg, polarization):
        N = d.shape[0]
        N_wl = wl.shape[0]
        dtype_c = torch.complex128

        wl_b = wl.view(1, -1).expand(N, -1)
        n_base = self.get_base_index(wl_b)
        n_layers = n_base * (1 + dn.unsqueeze(1))
        n_layer_0 = self.moth_eye_effective_n(n_layers[..., 0], f)
        n_layers = torch.cat([n_layer_0.unsqueeze(-1), n_layers[..., 1:]], dim=-1)
        n_layers = n_layers.to(dtype_c)

        M = torch.eye(2, dtype=dtype_c, device=self.device).view(1, 1, 2, 2).expand(N, N_wl, -1, -1).clone()
        k0 = (2 * np.pi / wl_b).to(dtype_c)
        theta = torch.deg2rad(torch.tensor(float(theta_deg), device=self.device, dtype=torch.float64))
        sin_theta0 = torch.sin(theta)
        cos_theta0 = torch.cos(theta).to(dtype_c)

        for i in range(5):
            ni = n_layers[..., i]
            di = d[:, i].view(-1, 1)
            sin_thetai = (self.config.AIR_N / ni) * sin_theta0
            sin_thetai = torch.clamp(sin_thetai.real, -0.99999, 0.99999)
            cos_thetai = torch.sqrt(1 - sin_thetai ** 2).to(dtype_c)
            delta = k0 * ni * di * cos_thetai

            cos_d = torch.cos(delta)
            sin_d = torch.sin(delta)

            # R2-minor-1: 按偏振取特征导纳
            #   s (TE): eta = n cos(theta)   —— 与原实现一致
            #   p (TM): eta = n / cos(theta)
            eta_i = ni / cos_thetai if polarization == 'p' else ni * cos_thetai
            m11 = cos_d
            m12 = 1j * sin_d / eta_i
            m21 = 1j * eta_i * sin_d
            m22 = cos_d

            Mi = torch.stack([torch.stack([m11, m12], dim=-1), torch.stack([m21, m22], dim=-1)], dim=-2)
            M = torch.matmul(M, Mi)

        ns = torch.tensor(self.config.SUBSTRATE_N, dtype=dtype_c, device=self.device)
        sin_thetas = torch.clamp(((self.config.AIR_N / ns) * sin_theta0).real, -0.99999, 0.99999)
        cos_thetas = torch.sqrt(1 - sin_thetas ** 2).to(dtype_c)
        if polarization == 'p':
            eta_s = ns / cos_thetas
            eta_0 = (self.config.AIR_N / cos_theta0).to(dtype_c)
        else:
            eta_s = ns * cos_thetas
            eta_0 = (self.config.AIR_N * cos_theta0).to(dtype_c)

        m11, m12 = M[..., 0, 0], M[..., 0, 1]
        m21, m22 = M[..., 1, 0], M[..., 1, 1]
        B = m11 + m12 * eta_s
        C = m21 + m22 * eta_s
        r = (eta_0 * B - C) / (eta_0 * B + C + 1e-12)
        return torch.clamp(torch.abs(r) ** 2, 0, 1)


# ============================================================
# 数据生成器
# ============================================================

class DataGenerator:
    def __init__(self, config: Config, physics: PhysicsModel):
        self.config = config
        self.physics = physics
        self.device = torch.device(config.DEVICE)

    def generate(self, seed: int = 42) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        true_d = torch.tensor([
            self.config.DESIGN_WAVELENGTH / (4 * 1.45),
            self.config.DESIGN_WAVELENGTH / (4 * 1.95),
            self.config.DESIGN_WAVELENGTH / (4 * 1.45),
            self.config.DESIGN_WAVELENGTH / (4 * 2.05),
            self.config.DESIGN_WAVELENGTH / (4 * 1.38)
        ], device=self.device)
        true_dn = torch.tensor([0.01, -0.01, 0.0, 0.01, -0.005], device=self.device)
        true_f = torch.tensor([self.config.TRUE_F_ULD], device=self.device)
        wl = torch.linspace(*self.config.WAVELENGTH_RANGE, self.config.N_WAVELENGTHS, device=self.device)

        with torch.no_grad():
            R_true_0 = self.physics.tmm_reflectance(true_d.unsqueeze(0), true_dn.unsqueeze(0), true_f.view(1, 1), wl,
                                                    0.0).squeeze(0)
            R_true_30 = self.physics.tmm_reflectance(true_d.unsqueeze(0), true_dn.unsqueeze(0), true_f.view(1, 1), wl,
                                                     30.0).squeeze(0)

        # 可复现性: 噪声在 CPU 端由固定种子流生成后再搬到计算设备，
        # 同一 seed 在 CPU 与 GPU 上给出完全相同的观测数据
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        noise_0 = self.config.SIGMA_OBS * torch.randn(R_true_0.shape, generator=gen).to(R_true_0.device)
        noise_30 = self.config.SIGMA_OBS * torch.randn(R_true_30.shape, generator=gen).to(R_true_30.device)
        R_obs_0 = R_true_0 + noise_0
        R_obs_30 = R_true_30 + noise_30

        return {
            'true_params': (true_d, true_dn, true_f),
            'wavelengths': wl,
            'R_true': (R_true_0, R_true_30),
            'R_obs': (R_obs_0, R_obs_30),
            'center_idx': torch.argmin(torch.abs(wl - self.config.DESIGN_WAVELENGTH)).item()
        }


# ============================================================
# ASMC and Gaussian VA methods
# ============================================================

class SMCMethod:
    def __init__(self, config: Config, physics: PhysicsModel):
        self.config = config
        self.physics = physics
        self.device = torch.device(config.DEVICE)
        self.lower, self.upper = config.parameter_box()

    def log_likelihood_batch(self, particles, R_obs_0, R_obs_30, wl):
        """R1-1 / R2-4: 标准高斯对数似然。

        删除了原实现中 632.8 nm 波长的十倍加权与 1/N 归一化：前者没有
        统计依据（审稿人 1 第 1 条），后者把似然压低 N 倍、等效放大噪声
        sqrt(N) 倍。两角度平方和除以 sigma^2，与 Gaussian VA 的数据项
        完全一致（同一目标函数）。
        """
        particles_tensor = torch.tensor(particles, device=self.device, dtype=torch.float64)
        d = particles_tensor[:, :5]
        dn = particles_tensor[:, 5:10]
        f = particles_tensor[:, 10:11]
        with torch.no_grad():
            R_pred_0 = self.physics.tmm_reflectance(d, dn, f, wl, 0.0)
            R_pred_30 = self.physics.tmm_reflectance(d, dn, f, wl, 30.0)
        err_0 = torch.sum((R_obs_0 - R_pred_0) ** 2, dim=1)
        err_30 = torch.sum((R_obs_30 - R_pred_30) ** 2, dim=1)
        ll = -0.5 * (err_0 + err_30) / (self.config.SIGMA_OBS ** 2)
        return ll.cpu().numpy()

    def in_bounds(self, particles):
        """R2-3: 先验可行域（均匀盒）判定，与 Gaussian VA 共用 Config 定义。"""
        return np.all((particles >= self.lower) & (particles <= self.upper), axis=-1)

    def _initial_particles(self, count):
        """Draw from the mixture proposal with initial weights p(theta)/q(theta)."""
        center, scale = self.config.starting_parameters()
        a = (self.lower - center) / scale
        b = (self.upper - center) / scale
        uniform_fraction = 0.75
        particles = np.random.uniform(self.lower, self.upper, size=(count, center.size))
        centered = np.random.random(count) >= uniform_fraction
        if centered.any():
            particles[centered] = truncnorm.rvs(
                a, b, loc=center, scale=scale, size=(centered.sum(), center.size))
        log_prior = -np.log(self.upper - self.lower).sum()
        log_centered = truncnorm.logpdf(particles, a, b, loc=center, scale=scale).sum(axis=1)
        log_proposal = np.logaddexp(np.log(uniform_fraction) + log_prior,
                                    np.log1p(-uniform_fraction) + log_centered)
        return particles, log_prior - log_proposal

    def _sync_device(self):
        # 运行时间统计需要同步异步 GPU 队列（审稿人对 runtime 的质疑）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def run(self, data: Dict, seed: int = 42) -> Dict:
        """自适应退火 SMC。初始重要性权重以均匀盒先验为目标。"""

        torch.manual_seed(seed)
        np.random.seed(seed)
        R_obs_0, R_obs_30 = data['R_obs']
        wl = data['wavelengths']

        N = self.config.N_PARTICLES_SMC
        T_max = self.config.T_STAGES_SMC
        ess_target = self.config.ESS_THRESHOLD_FRAC * N

        particles, logw = self._initial_particles(N)
        initial_weights = np.exp(logw - logw.max())
        initial_weights /= initial_weights.sum()
        if 1.0 / np.sum(initial_weights ** 2) < ess_target:
            particles = particles[np.random.choice(N, size=N, p=initial_weights)]
            logw.fill(0.0)
        beta = 0.0
        betas_history = []
        ess_history = []
        acceptance_history = []
        ll_current = self.log_likelihood_batch(particles, R_obs_0, R_obs_30, wl)

        def coord_mh_sweep(n_sweeps):
            """对盒内的对称提议使用当前退火似然接受率。"""
            acc_all = []
            for _ in range(n_sweeps):
                for k in range(11):
                    step = self.config.JITTER_SCALE * particles[:, k].std() + 1e-12
                    proposal = particles.copy()
                    proposal[:, k] = particles[:, k] + step * np.random.randn(N)
                    inside = self.in_bounds(proposal)
                    ll_prop = np.full(N, -np.inf)
                    if np.any(inside):
                        ll_prop[inside] = self.log_likelihood_batch(proposal[inside], R_obs_0, R_obs_30, wl)
                    log_alpha = beta * (ll_prop - ll_current)
                    accept = (np.log(np.random.rand(N)) < log_alpha) & inside
                    particles[accept] = proposal[accept]
                    ll_current[accept] = ll_prop[accept]
                    acc_all.append(float(accept.mean()))
            return float(np.mean(acc_all)) if acc_all else float('nan')

        self._sync_device()
        start_time = time.time()

        stage = 0
        while beta < 1.0 - 1e-9 and stage < T_max:
            # --- 自适应退火：二分搜索本级最大温度增量 d_beta ----------
            lo, hi = 1e-12, 1.0 - beta
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                test = logw + mid * ll_current
                test = test - test.max()
                w_test = np.exp(test)
                w_test /= w_test.sum()
                if 1.0 / np.sum(w_test ** 2) >= ess_target:
                    lo = mid
                else:
                    hi = mid
            d_beta = min(lo, 1.0 - beta)
            beta = beta + d_beta
            betas_history.append(beta)

            logw += d_beta * ll_current
            logw -= logw.max()
            w = np.exp(logw)
            w /= np.sum(w)
            ess = 1.0 / np.sum(w ** 2)
            ess_history.append(ess)

            # 每级重采样恢复 ESS（增量权重 ESS 已被自适应调度限制在
            # ess_target 以上，重采样损失可控）
            idx = np.random.choice(N, size=N, p=w)
            particles = particles[idx]
            ll_current = ll_current[idx]
            logw.fill(0.0)

            # --- R2-6: 逐坐标 Metropolis-Hastings 移动核 ---
            mh_rate = coord_mh_sweep(self.config.MH_SWEEPS)
            acceptance_history.append(mh_rate)

            if (stage + 1) % 5 == 0:
                print(f"  SMC进度: stage {stage + 1} | beta: {beta:.4f} | ESS: {ess:.0f}/{N}"
                      f" | MH接受率: {mh_rate:.2f}")
            stage += 1

        # --- beta=1 后的精修扫描（改善末期混合） ---
        if beta >= 1.0 - 1e-9 and self.config.FINAL_SWEEPS > 0:
            mh_rate = coord_mh_sweep(self.config.FINAL_SWEEPS)
            acceptance_history.append(mh_rate)

        # --- R2-6: 终估按最终权重重采样为等权后验样本 ---
        w_final = np.exp(logw - logw.max())
        w_final /= w_final.sum()
        final_idx = np.random.choice(N, size=N, p=w_final)
        particles = particles[final_idx]

        self._sync_device()
        runtime = time.time() - start_time
        return {'particles': particles,
                'betas_history': betas_history,
                'ess_history': ess_history,
                'acceptance_history': acceptance_history,
                'runtime': runtime}


class GaussianVA(nn.Module):
    """Bounded diagonal Gaussian approximation with a uniform-box prior."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.DEVICE)
        lower, upper = torch.as_tensor(config.parameter_box(), dtype=torch.float64, device=self.device)
        center, physical_std = torch.as_tensor(
            np.array(config.starting_parameters()), dtype=torch.float64, device=self.device)
        width = upper - lower
        fraction = (center - lower) / width
        if torch.any((fraction <= 0) | (fraction >= 1)):
            raise ValueError("Gaussian VA initialization must be inside the prior box.")
        self.register_buffer('lower', lower)
        self.register_buffer('width', width)
        self._sigmoid = SigmoidTransform()

        # Keep the latent coordinates on the physical scale near the box midpoint.
        self.mu = nn.Parameter(torch.logit(fraction) * width / 4.0)
        self.log_std = nn.Parameter(torch.log(physical_std / (4.0 * fraction * (1.0 - fraction))))

    def _latent_samples(self, n_samples):
        return Normal(self.mu, self.log_std.exp()).rsample((n_samples,))

    def _to_box(self, latent):
        logits = 4.0 * latent / self.width
        fraction = self._sigmoid(logits)
        theta = self.lower + self.width * fraction
        log_jacobian = (math.log(4.0) + self._sigmoid.log_abs_det_jacobian(logits, fraction)).sum(dim=-1)
        return theta, log_jacobian

    def _sample_with_kl(self, n_samples):
        theta, log_jacobian = self._to_box(self._latent_samples(n_samples))
        latent_entropy = self.log_std.sum() + 0.5 * self.mu.numel() * (1.0 + math.log(2.0 * math.pi))
        kl = self.width.log().sum() - latent_entropy - log_jacobian.mean()
        return theta.split((5, 5, 1), dim=-1), kl

    def sample(self, n_samples=1):
        theta, _ = self._to_box(self._latent_samples(n_samples))
        return theta.split((5, 5, 1), dim=-1)

    def run(self, data: Dict, physics: PhysicsModel, seed: int = 42) -> Dict:
        """Minimize expected Gaussian NLL plus KL to the same prior as ASMC."""
        torch.manual_seed(seed)
        R_obs_0, R_obs_30 = data['R_obs']
        wl = data['wavelengths']
        optimizer = optim.Adam([
            {'params': [p for p in self.parameters() if p is not self.log_std],
             'lr': self.config.LR_VA},
            {'params': [self.log_std], 'lr': self.config.LR_VA * self.config.LR_STD_MULTIPLIER},
        ])
        start_time = time.time()
        loss_history = []
        for epoch in range(self.config.EPOCHS_VA):
            optimizer.zero_grad()
            (d_s, dn_s, f_s), kl = self._sample_with_kl(self.config.N_SAMPLES_VA)

            R_p0 = physics.tmm_reflectance(d_s, dn_s, f_s, wl, 0.0)
            R_p30 = physics.tmm_reflectance(d_s, dn_s, f_s, wl, 30.0)
            err_0 = torch.sum((R_p0 - R_obs_0.unsqueeze(0)) ** 2, dim=1)
            err_30 = torch.sum((R_p30 - R_obs_30.unsqueeze(0)) ** 2, dim=1)
            loss_nll = 0.5 * torch.mean(err_0 + err_30) / (self.config.SIGMA_OBS ** 2)
            loss = loss_nll + self.config.KL_WEIGHT * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                self.log_std.clamp_(min=self.config.STD_LOG_MIN)
            loss_history.append(loss.item())
            if (epoch + 1) % 500 == 0:
                print(f"  {type(self).__name__} progress: {epoch + 1}/{self.config.EPOCHS_VA} | "
                      f"loss: {loss.item():.6f}")
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        with torch.no_grad():
            d_post, dn_post, f_post = self.sample(4000)
        return {'particles': torch.cat([d_post, dn_post, f_post], dim=1).cpu().numpy(),
                'loss_history': loss_history,
                'runtime': time.time() - start_time}


class GaussianVAFullCov(GaussianVA):
    """Full-covariance variant with the same bounded transform and KL."""

    def __init__(self, config: Config):
        super().__init__(config)
        size = self.mu.numel()
        indices = torch.tril_indices(size, size, offset=-1, device=self.device)
        self.register_buffer('lower_indices', indices)
        self.chol_lower = nn.Parameter(self.mu.new_zeros(indices.shape[1]))

    def _scale_tril(self):
        scale = self.log_std.exp()
        rows, cols = self.lower_indices
        return torch.diag(scale).index_put((rows, cols), self.chol_lower * scale[rows])

    def _latent_samples(self, n_samples):
        eps = torch.randn(n_samples, self.mu.numel(), device=self.mu.device, dtype=self.mu.dtype)
        return self.mu + eps @ self._scale_tril().T

    @torch.no_grad()
    def covariance(self):
        """Sample estimate of covariance in the bounded physical coordinates."""
        generator = torch.Generator(device=self.mu.device).manual_seed(0)
        eps = torch.randn(4000, self.mu.numel(), generator=generator,
                          device=self.mu.device, dtype=self.mu.dtype)
        theta, _ = self._to_box(self.mu + eps @ self._scale_tril().T)
        return torch.cov(theta.T).cpu().numpy()


# ============================================================
# 独立参照解 (R2-8)
# ============================================================

class ReferenceMCMC:
    """R2-8: 独立随机游走 Metropolis 参照采样器。

    与两个被评估方法使用同一似然（SMCMethod.log_likelihood_batch）与
    同一有界均匀先验（拒绝越界提议）。单一长链，burn-in 期间按
    Robbins-Monro 自适应步长（目标接受率 0.234），随后按 2.38/sqrt(d)
    规则用 burn-in 经验散布重标定提议协方差。作为"独立验证解"用于
    核对两个近似方法的后验均值与宽度。
    """

    def __init__(self, config: Config, physics: PhysicsModel):
        self.config = config
        self.physics = physics
        self.device = torch.device(config.DEVICE)

    def run(self, data: Dict, seed: int = 42) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = self.config
        R_obs_0, R_obs_30 = data['R_obs']
        wl = data['wavelengths']
        n_steps = int(cfg.REF_MCMC_STEPS)
        burnin = min(int(cfg.REF_MCMC_BURNIN), max(1, n_steps // 2))
        smc = SMCMethod(cfg, self.physics)

        d_design = np.array([
            cfg.DESIGN_WAVELENGTH / (4 * 1.45), cfg.DESIGN_WAVELENGTH / (4 * 1.95),
            cfg.DESIGN_WAVELENGTH / (4 * 1.45), cfg.DESIGN_WAVELENGTH / (4 * 2.05),
            cfg.DESIGN_WAVELENGTH / (4 * 1.38)
        ]) * cfg.INIT_D_SCALE
        theta = np.concatenate([d_design, np.zeros(5), [cfg.TRUE_F_ULD]])
        prop_sd = np.array([1.0] * 5 + [0.002] * 5 + [0.005])
        ll = float(smc.log_likelihood_batch(theta[None, :], R_obs_0, R_obs_30, wl)[0])

        accepted = 0
        window = 0
        samples = []
        self._sync()
        start_time = time.time()
        for step in range(n_steps):
            proposal = theta + prop_sd * np.random.randn(11)
            if not smc.in_bounds(proposal[None, :])[0]:
                samples.append(theta.copy())
                continue
            ll_prop = float(smc.log_likelihood_batch(proposal[None, :], R_obs_0, R_obs_30, wl)[0])
            if np.log(np.random.rand()) < (ll_prop - ll):
                theta, ll = proposal, ll_prop
                accepted += 1
                window += 1
            samples.append(theta.copy())
            if step < burnin and (step + 1) % 100 == 0:
                prop_sd *= float(np.exp(window / 100.0 - 0.234))
                window = 0
        burn_samples = np.asarray(samples[:burnin])
        emp = burn_samples.std(axis=0)
        emp = np.where(emp < 1e-12, prop_sd, emp)
        prop_sd = (2.38 / np.sqrt(11)) * emp
        # 用重标定后的提议继续采样
        for step in range(n_steps):
            proposal = theta + prop_sd * np.random.randn(11)
            if not smc.in_bounds(proposal[None, :])[0]:
                samples.append(theta.copy())
                continue
            ll_prop = float(smc.log_likelihood_batch(proposal[None, :], R_obs_0, R_obs_30, wl)[0])
            if np.log(np.random.rand()) < (ll_prop - ll):
                theta, ll = proposal, ll_prop
                accepted += 1
                window += 1
            samples.append(theta.copy())
        self._sync()
        particles = np.asarray(samples)[burnin:]
        return {'particles': particles,
                'acceptance_rate': accepted / max(1, 2 * n_steps),
                'runtime': time.time() - start_time}

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()


# ============================================================
# 评估器
# ============================================================

class Evaluator:
    @staticmethod
    def evaluate(particles: np.ndarray, true_params: Tuple, method_name: str) -> Dict:
        """R2-9 / R1-2: 全部 11 个参数的误差与 2-sigma 覆盖率。"""
        true_d, true_dn, true_f = [p.cpu().numpy() if torch.is_tensor(p) else p for p in true_params]
        d_mean = particles[:, :5].mean(axis=0)
        d_std = particles[:, :5].std(axis=0)
        d_err = np.abs(d_mean - true_d) / true_d * 100
        dn_mean = particles[:, 5:10].mean(axis=0)
        dn_std = particles[:, 5:10].std(axis=0)
        dn_err = np.abs(dn_mean - true_dn)
        f_mean = particles[:, 10].mean()
        f_std = particles[:, 10].std()
        f_err = np.abs(f_mean - true_f) / true_f * 100

        # R1-2: 经验 2-sigma 覆盖率（真值是否落在均值 +/- 2*std 内）
        d_covered = np.abs(d_mean - true_d) <= 2.0 * d_std
        f_covered = np.abs(f_mean - true_f) <= 2.0 * f_std

        return {
            'method': method_name,
            'thickness_rmse': np.sqrt(np.mean((d_mean - true_d) ** 2)),
            'thickness_mae': np.mean(np.abs(d_mean - true_d)),
            'thickness_rel_err': np.mean(d_err),
            'thickness_layer_err': d_err.tolist(),
            'dn_abs_err_mean': float(np.mean(dn_err)),
            'dn_abs_err': dn_err.tolist(),
            'porosity_rel_err': f_err.item(),
            'avg_std': np.mean(d_std),
            'd_mean': d_mean.tolist(),
            'd_std': d_std.tolist(),
            'dn_mean': dn_mean.tolist(),
            'dn_std': dn_std.tolist(),
            'f_mean': float(f_mean),
            'f_std': float(f_std),
            'thickness_coverage_2sigma': float(np.mean(d_covered)),
            'porosity_coverage_2sigma': float(f_covered)
        }

    @staticmethod
    def compute_spectral_rmse(particles: np.ndarray, physics: PhysicsModel, data: Dict,
                              n_samples: int = 200, seed: int = None) -> Dict:
        """R2-9: 两角度光谱 RMSE（带种子的确定性子采样，可复现）。"""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(particles), min(n_samples, len(particles)), replace=False)
        ps = torch.tensor(particles[idx], device=physics.device, dtype=torch.float64)
        with torch.no_grad():
            R0 = physics.tmm_reflectance(ps[:, :5], ps[:, 5:10], ps[:, 10:], data['wavelengths'], 0.0)
            R30 = physics.tmm_reflectance(ps[:, :5], ps[:, 5:10], ps[:, 10:], data['wavelengths'], 30.0)
        mu0 = R0.mean(0).cpu().numpy()
        mu30 = R30.mean(0).cpu().numpy()
        obs0 = data['R_obs'][0].cpu().numpy()
        obs30 = data['R_obs'][1].cpu().numpy()
        rmse0 = float(np.sqrt(np.mean((mu0 - obs0) ** 2)))
        rmse30 = float(np.sqrt(np.mean((mu30 - obs30) ** 2)))
        sigma = float(physics.config.SIGMA_OBS)
        return {'rmse_0': rmse0, 'rmse_30': rmse30,
                'rmse_0_over_sigma': rmse0 / sigma, 'rmse_30_over_sigma': rmse30 / sigma}

    @staticmethod
    def print_detailed_metrics(sigma, res_smc, res_pinn, data, runtime_smc, runtime_pinn,
                               physics=None, seed=None):
        eval_smc = Evaluator.evaluate(res_smc['particles'], data['true_params'], 'ASMC')
        eval_pinn = Evaluator.evaluate(res_pinn['particles'], data['true_params'], 'Gaussian VA')

        # R2-9: 用与被评估方法一致的物理模型计算两角度 RMSE（原实现新建
        # 了默认 Config 的 PhysicsModel，噪声水平与当前实验不符）
        if physics is None:
            physics = PhysicsModel(Config())
        rmse_smc = Evaluator.compute_spectral_rmse(res_smc['particles'], physics, data, seed=seed)
        rmse_pinn = Evaluator.compute_spectral_rmse(res_pinn['particles'], physics, data, seed=seed)

        print("\n" + "=" * 80)
        print(f"噪声水平 sigma={sigma} - 核心对比数据")
        print("=" * 80)
        print("\n【基础性能对比】")
        print(f"{'指标':<24} {'ASMC':<15} {'Gaussian VA':<15} {'优势方':<10}")
        print("-" * 64)
        rows = [
            ('光谱RMSE (0°)', rmse_smc['rmse_0'], rmse_pinn['rmse_0']),
            ('光谱RMSE (30°)', rmse_smc['rmse_30'], rmse_pinn['rmse_30']),
            ('厚度相对误差(%)', eval_smc['thickness_rel_err'], eval_pinn['thickness_rel_err']),
            ('Δn绝对误差均值', eval_smc['dn_abs_err_mean'], eval_pinn['dn_abs_err_mean']),
            ('孔隙率相对误差(%)', eval_smc['porosity_rel_err'], eval_pinn['porosity_rel_err']),
            ('运行时间(s)', runtime_smc, runtime_pinn),
        ]
        for name, v_smc, v_pinn in rows:
            better = 'ASMC' if v_smc < v_pinn else 'Gaussian VA'
            fmt = '{:.2e}' if abs(v_smc) < 1e-2 else '{:.3f}'
            print(f"{name:<24} {fmt.format(v_smc):<15} {fmt.format(v_pinn):<15} {better:<10}")

        print("\n【光谱误差与注入噪声之比（达到噪声底应接近1）】")
        print(f"{'指标':<24} {'ASMC':<15} {'Gaussian VA':<15}")
        print("-" * 54)
        print(f"{'RMSE(0°)/sigma':<24} {rmse_smc['rmse_0_over_sigma']:<15.2f} "
              f"{rmse_pinn['rmse_0_over_sigma']:<15.2f}")
        print(f"{'RMSE(30°)/sigma':<24} {rmse_smc['rmse_30_over_sigma']:<15.2f} "
              f"{rmse_pinn['rmse_30_over_sigma']:<15.2f}")

        print("\n【逐层厚度相对误差(%)】")
        layers = ['L1: SiO2-ULD', 'L2: HfO2', 'L3: SiO2', 'L4: Ta2O5', 'L5: MgF2']
        print(f"{'层名':<15} {'ASMC误差(%)':<15} {'Gaussian VA误差(%)':<15}")
        print("-" * 45)
        for i, layer in enumerate(layers):
            print(f"{layer:<15} {eval_smc['thickness_layer_err'][i]:<15.2f} "
                  f"{eval_pinn['thickness_layer_err'][i]:<15.2f}")

        print("\n【折射率漂移 Δn 逐层绝对误差 (R2-9)】")
        print(f"{'层名':<15} {'ASMC':<15} {'Gaussian VA':<15}")
        print("-" * 45)
        for i, layer in enumerate(layers):
            print(f"{layer:<15} {eval_smc['dn_abs_err'][i]:<15.4f} {eval_pinn['dn_abs_err'][i]:<15.4f}")

        print("\n【后验分布统计】")
        print(f"{'指标':<24} {'ASMC':<15} {'Gaussian VA':<15}")
        print("-" * 54)
        print(f"{'厚度平均标准差(nm)':<24} {eval_smc['avg_std']:.2f} {eval_pinn['avg_std']:.2f}")
        print(f"{'孔隙率均值':<24} {eval_smc['f_mean']:.4f} {eval_pinn['f_mean']:.4f}")
        print(f"{'孔隙率标准差':<24} {eval_smc['f_std']:.4f} {eval_pinn['f_std']:.4f}")
        print(f"{'厚度2σ覆盖率':<24} {eval_smc['thickness_coverage_2sigma']:.2f} "
              f"{eval_pinn['thickness_coverage_2sigma']:.2f}")
        print(f"{'孔隙率2σ覆盖率':<24} {eval_smc['porosity_coverage_2sigma']:.2f} "
              f"{eval_pinn['porosity_coverage_2sigma']:.2f}")

        print("\n【ASMC特有指标】")
        final_ess = res_smc['ess_history'][-1] if res_smc['ess_history'] else 0
        final_beta = res_smc['betas_history'][-1] if res_smc.get('betas_history') else 0
        mh_rate = np.mean(res_smc['acceptance_history']) if res_smc.get('acceptance_history') else float('nan')
        print(f"最终有效样本数(ESS): {final_ess:.0f}/{physics.config.N_PARTICLES_SMC}")
        print(f"自适应退火级数: {len(res_smc['betas_history'])} | 最终 beta: {final_beta:.4f}")
        print(f"MH平均接受率: {mh_rate:.2f}")

        print("\n【Gaussian VA特有指标】")
        final_loss = res_pinn['loss_history'][-1] if res_pinn['loss_history'] else 0
        print(f"最终训练损失: {final_loss:.6f}")
        print("\n" + "=" * 80 + "\n")

        return {
            'sigma': sigma,
            'spectral_rmse_smc': rmse_smc['rmse_0'],
            'spectral_rmse_pinn': rmse_pinn['rmse_0'],
            'spectral_rmse30_smc': rmse_smc['rmse_30'],
            'spectral_rmse30_pinn': rmse_pinn['rmse_30'],
            'thickness_rel_err_smc': eval_smc['thickness_rel_err'],
            'thickness_rel_err_pinn': eval_pinn['thickness_rel_err'],
            'dn_abs_err_smc': eval_smc['dn_abs_err_mean'],
            'dn_abs_err_pinn': eval_pinn['dn_abs_err_mean'],
            'porosity_rel_err_smc': eval_smc['porosity_rel_err'],
            'porosity_rel_err_pinn': eval_pinn['porosity_rel_err'],
            'coverage_smc': eval_smc['thickness_coverage_2sigma'],
            'coverage_pinn': eval_pinn['thickness_coverage_2sigma'],
            'runtime_smc': runtime_smc,
            'runtime_pinn': runtime_pinn
        }


# ============================================================
# 审稿人回应分析工具
# ============================================================

def validate_forward_model(sigma: float = 1e-4, seed: int = 2024) -> Dict:
    """R2-1: 前向算子物理自检。

    把全部厚度置零，TMM 必须退化为裸基底的单界面菲涅尔反射率。
    原实现（基底导纳放在入射介质位置）给出 R=0，本实现应与解析值
    在 1e-6 内一致。
    """
    config = Config()
    config.SIGMA_OBS = sigma
    physics = PhysicsModel(config)
    wl = torch.linspace(
        *config.WAVELENGTH_RANGE, config.N_WAVELENGTHS,
        device=physics.device, dtype=torch.float64,
    )
    idx = int(torch.argmin(torch.abs(wl - config.DESIGN_WAVELENGTH)).item())

    def fresnel(n0, ns, theta_deg, polarization):
        t = math.radians(theta_deg)
        c0 = math.cos(t)
        s_s = n0 * math.sin(t) / ns
        c_s = math.sqrt(max(0.0, 1.0 - s_s ** 2))
        r_s = (n0 * c0 - ns * c_s) / (n0 * c0 + ns * c_s)
        r_p = (n0 * c_s - ns * c0) / (n0 * c_s + ns * c0)
        if polarization == 'unpolarized':
            return 0.5 * (r_s ** 2 + r_p ** 2)
        return r_p ** 2 if polarization == 'p' else r_s ** 2

    with torch.no_grad():
        z_d = torch.zeros((1, config.N_LAYERS), dtype=torch.float64, device=physics.device)
        z_dn = torch.zeros_like(z_d)
        f_vec = torch.tensor([[config.TRUE_F_ULD]], dtype=torch.float64, device=physics.device)
        R0 = physics.tmm_reflectance(z_d, z_dn, f_vec, wl, 0.0)
        R30 = physics.tmm_reflectance(z_d, z_dn, f_vec, wl, 30.0)
    got0 = float(R0[0, idx].item())
    got30 = float(R30[0, idx].item())
    exp0 = fresnel(config.AIR_N, config.SUBSTRATE_N, 0.0, config.POLARIZATION)
    exp30 = fresnel(config.AIR_N, config.SUBSTRATE_N, 30.0, config.POLARIZATION)
    print("=" * 80)
    print("R2-1 前向算子自检（全厚度置零 = 裸基底菲涅尔极限）")
    print("=" * 80)
    print(f"  0°  : 实现值 {got0:.6f} | 解析值 {exp0:.6f} | 偏差 {abs(got0 - exp0):.2e}")
    print(f"  30° : 实现值 {got30:.6f} | 解析值 {exp30:.6f} | 偏差 {abs(got30 - exp30):.2e}")
    print(f"  偏振: {config.POLARIZATION}")
    results = {'R0': got0, 'R0_expected': exp0, 'R30': got30, 'R30_expected': exp30}
    return results


def print_prior_specification(config: Config = None):
    """R2-3: 先验与起始值的显式说明（区分二者，确认两法共用）。"""
    cfg = config or Config()
    print("=" * 80)
    print("R2-3 先验规范与起始值")
    print("=" * 80)
    print(f"  先验（两法共用，均匀盒）:")
    print(f"    d  ~ U{cfg.BOUNDS_D} (nm)")
    print(f"    dn ~ U{cfg.BOUNDS_DN}")
    print(f"    f  ~ U{cfg.BOUNDS_F}")
    center, _ = cfg.starting_parameters()
    print("  设计中心:")
    print(f"    设计厚度 x {cfg.INIT_D_SCALE}, dn=0, f={center[-1]}")
    print(f"    物理尺度: d={cfg.INIT_SIGMA_D} nm, dn={cfg.INIT_SIGMA_DN}, f={cfg.INIT_SIGMA_F}")
    print("  ASMC 以均匀盒和设计中心截断高斯为混合提议，按先验/提议密度初始化权重。")
    print("  Gaussian VA 用 sigmoid 映射进入均匀盒，KL 项包含变换雅可比。")


def analyze_posterior_correlations(particles: np.ndarray) -> Dict:
    """R1-4: 后验相关性定量分析。"""
    x = np.asarray(particles, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.asarray(np.corrcoef(x, rowvar=False)))
    np.fill_diagonal(corr, 1.0)
    iu = np.triu_indices(corr.shape[0], k=1)
    pairs = sorted([(PARAM_LABELS[i], PARAM_LABELS[j], float(corr[i, j]))
                    for i, j in zip(*iu)], key=lambda t: -abs(t[2]))
    print("\n【R1-4 后验相关性】最强关联参数对:")
    for a, b, v in pairs[:6]:
        print(f"    corr({a:>10s}, {b:>10s}) = {v:+.3f}")
    print(f"    d1-f 相关性 = {float(corr[0, 10]):+.3f}")
    return {'corr': corr, 'top_pairs': pairs[:6], 'd1_f_corr': float(corr[0, 10])}


def porosity_forward_sensitivity(sigma: float = 1e-4, seed: int = 2024, n_grid: int = 41):
    """R1-6: 孔隙率前向敏感性分析。

    固定其余参数为真值，扫描 f 遍历整个先验 [0.30, 0.50]，给出两角度
    光谱相对真值谱的最大 RMSE 与 n_surface(f) 的变化范围，以及"光谱
    变化低于噪声 sigma 因而对 f 不可分辨"的区间宽度。
    """
    config = Config()
    config.SIGMA_OBS = sigma
    physics = PhysicsModel(config)
    data = DataGenerator(config, physics).generate(seed)
    true_d, true_dn, true_f = data['true_params']
    wl = data['wavelengths']
    R_true_0, R_true_30 = data['R_true']

    f_grid = np.linspace(config.BOUNDS_F[0], config.BOUNDS_F[1], n_grid)
    with torch.no_grad():
        d_t = torch.tensor(np.tile(true_d.cpu().numpy(), (n_grid, 1)), dtype=torch.float64)
        dn_t = torch.tensor(np.tile(true_dn.cpu().numpy(), (n_grid, 1)), dtype=torch.float64)
        f_t = torch.tensor(f_grid.reshape(-1, 1), dtype=torch.float64)
        R0 = physics.tmm_reflectance(d_t, dn_t, f_t, wl, 0.0).cpu().numpy()
        R30 = physics.tmm_reflectance(d_t, dn_t, f_t, wl, 30.0).cpu().numpy()
        n_uld = physics.get_base_index(wl.view(1, -1))[..., 0]
        n_surface = physics.moth_eye_effective_n(n_uld, f_t).cpu().numpy()

    ref0 = R_true_0.cpu().numpy()
    ref30 = R_true_30.cpu().numpy()
    rmse = np.array([np.sqrt(np.mean((np.concatenate([R0[k] - ref0, R30[k] - ref30])) ** 2))
                     for k in range(n_grid)])
    indistinguishable = rmse <= float(sigma)
    width = float(f_grid[indistinguishable].max() - f_grid[indistinguishable].min()) \
        if indistinguishable.any() else 0.0

    print("=" * 80)
    print(f"R1-6 孔隙率前向敏感性 (f 扫描 [{config.BOUNDS_F[0]}, {config.BOUNDS_F[1]}])")
    print("=" * 80)
    print(f"  两角度光谱最大 RMSE(f) = {rmse.max():.3e} (sigma = {sigma:g})")
    print(f"  n_surface 范围 = [{n_surface.min():.4f}, {n_surface.max():.4f}]")
    print(f"  与真值不可分辨的 f 区间宽度 = {width:.3f} "
          f"({width / (config.BOUNDS_F[1] - config.BOUNDS_F[0]) * 100:.0f}% 的先验范围)")
    return {'f_grid': f_grid, 'rmse': rmse, 'n_surface': n_surface,
            'indistinguishable_width': width}


def run_single_trial(sigma, seed, va_class=GaussianVA, **overrides) -> Dict:
    """单次 (sigma, seed) 完整试验，供多种子与敏感性分析复用。"""
    config = Config()
    config.SIGMA_OBS = float(sigma)
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"未知 Config 字段: {key}")
        setattr(config, key, value)
    physics = PhysicsModel(config)
    data = DataGenerator(config, physics).generate(seed)

    smc = SMCMethod(config, physics)
    res_smc = smc.run(data, seed)
    va = va_class(config).to(torch.device(config.DEVICE))
    res_va = va.run(data, physics, seed)

    ev_smc = Evaluator.evaluate(res_smc['particles'], data['true_params'], 'ASMC')
    ev_va = Evaluator.evaluate(res_va['particles'], data['true_params'], 'Gaussian VA')
    rm_smc = Evaluator.compute_spectral_rmse(res_smc['particles'], physics, data, seed=seed)
    rm_va = Evaluator.compute_spectral_rmse(res_va['particles'], physics, data, seed=seed)
    corr_smc = analyze_posterior_correlations(res_smc['particles']) if 'print_corr' in overrides else None
    return {
        'sigma': float(sigma), 'seed': int(seed),
        'smc': {**ev_smc, 'rmse_0': rm_smc['rmse_0'], 'rmse_30': rm_smc['rmse_30'],
                'runtime': res_smc['runtime'],
                'final_ess': res_smc['ess_history'][-1] if res_smc['ess_history'] else float('nan')},
        'va': {**ev_va, 'rmse_0': rm_va['rmse_0'], 'rmse_30': rm_va['rmse_30'],
               'runtime': res_va['runtime']}
    }


def print_seed_statistics(trials: List[Dict]):
    """R1-5: 多种子汇总（均值 ± 标准差）。"""
    print("\n" + "=" * 96)
    n_seeds = len({t['seed'] for t in trials})
    print(f"R1-5 多种子统计 (每噪声水平 {n_seeds} 个独立噪声实现)")
    print("=" * 96)
    header = f"{'指标':<28} {'ASMC (mean±std)':<26} {'Gaussian VA (mean±std)':<26}"
    for sigma in sorted({t['sigma'] for t in trials}):
        group = [t for t in trials if t['sigma'] == sigma]
        print(f"\n--- sigma = {sigma:g} ---")
        print(header)
        print("-" * 80)
        keys = [('厚度相对误差(%)', 'thickness_rel_err'),
                ('Δn绝对误差', 'dn_abs_err_mean'),
                ('孔隙率相对误差(%)', 'porosity_rel_err'),
                ('光谱RMSE 0°', 'rmse_0'),
                ('光谱RMSE 30°', 'rmse_30'),
                ('厚度平均std(nm)', 'avg_std'),
                ('厚度2σ覆盖率', 'thickness_coverage_2sigma'),
                ('运行时间(s)', 'runtime')]
        for label, key in keys:
            vals_s = np.array([t['smc'][key] for t in group], dtype=np.float64)
            vals_v = np.array([t['va'][key] for t in group], dtype=np.float64)
            fmt = '{:.2e}' if np.max(np.abs([vals_s.mean(), vals_v.mean()])) < 1e-2 else '{:.3f}'
            std_s = vals_s.std(ddof=1) if n_seeds > 1 else 0.0
            std_v = vals_v.std(ddof=1) if n_seeds > 1 else 0.0
            print(f"{label:<28} {fmt.format(vals_s.mean())} ± {fmt.format(std_s):<10} "
                  f"{fmt.format(vals_v.mean())} ± {fmt.format(std_v)}")


def sensitivity_asmc_settings(sigma: float = 5e-3, seed: int = 2024, **cheap):
    """R1-3: ASMC 数值配置敏感性（粒子数/级数上限/ESS阈值/提议尺度）。"""
    grid = [
        ('基准配置', {}),
        ('粒子数 1000', {'N_PARTICLES_SMC': 1000}),
        ('粒子数 8000', {'N_PARTICLES_SMC': 8000}),
        ('级数上限 25', {'T_STAGES_SMC': 25}),
        ('级数上限 100', {'T_STAGES_SMC': 100}),
        ('ESS阈值 0.25N', {'ESS_THRESHOLD_FRAC': 0.25}),
        ('ESS阈值 0.75N', {'ESS_THRESHOLD_FRAC': 0.75}),
        ('MH提议尺度 0.2', {'JITTER_SCALE': 0.2}),
        ('MH提议尺度 1.0', {'JITTER_SCALE': 1.0}),
    ]
    print("\n" + "=" * 96)
    print(f"R1-3 ASMC 数值配置敏感性 (sigma={sigma:g}, seed={seed})")
    print("=" * 96)
    print(f"{'配置':<20} {'厚度误差%':<12} {'孔隙率误差%':<14} {'RMSE 0°':<12} "
          f"{'平均std(nm)':<12} {'ESS':<10} {'MH接受率':<10}")
    print("-" * 90)
    for label, overrides in grid:
        merged = dict(cheap)
        merged.update(overrides)
        config = Config()
        config.SIGMA_OBS = sigma
        for k, v in merged.items():
            setattr(config, k, v)
        physics = PhysicsModel(config)
        data = DataGenerator(config, physics).generate(seed)
        res = SMCMethod(config, physics).run(data, seed)
        ev = Evaluator.evaluate(res['particles'], data['true_params'], 'ASMC')
        rm = Evaluator.compute_spectral_rmse(res['particles'], physics, data, seed=seed)
        ess = res['ess_history'][-1] if res['ess_history'] else float('nan')
        mh = np.mean(res['acceptance_history']) if res['acceptance_history'] else float('nan')
        print(f"{label:<20} {ev['thickness_rel_err']:<12.3f} {ev['porosity_rel_err']:<14.3f} "
              f"{rm['rmse_0']:<12.2e} {ev['avg_std']:<12.3f} {ess:<10.0f} {mh:<10.2f}")


def sensitivity_va_settings(sigma: float = 5e-3, seed: int = 2024, **cheap):
    """R2-8: Gaussian VA 设置敏感性（解释宽度为何/如何随噪声变化）。"""
    grid = [
        ('基准配置', {}),
        ('KL 权重 0（无正则）', {'KL_WEIGHT': 0.0}),
        ('KL 权重 10', {'KL_WEIGHT': 10.0}),
        ('训练轮数 1000', {'EPOCHS_VA': 1000}),
        ('训练轮数 10000', {'EPOCHS_VA': 10000}),
        ('学习率 5e-3', {'LR_VA': 5e-3}),
        ('每轮样本 64', {'N_SAMPLES_VA': 64}),
    ]
    print("\n" + "=" * 96)
    print(f"R2-8 Gaussian VA 设置敏感性 (sigma={sigma:g}, seed={seed})")
    print("=" * 96)
    print(f"{'配置':<22} {'厚度误差%':<12} {'孔隙率误差%':<14} {'RMSE 0°':<12} "
          f"{'平均std(nm)':<12} {'f_std':<10}")
    print("-" * 82)
    for label, overrides in grid:
        merged = dict(cheap)
        merged.update(overrides)
        config = Config()
        config.SIGMA_OBS = sigma
        for k, v in merged.items():
            setattr(config, k, v)
        physics = PhysicsModel(config)
        data = DataGenerator(config, physics).generate(seed)
        va = GaussianVA(config).to(torch.device(config.DEVICE))
        res = va.run(data, physics, seed)
        ev = Evaluator.evaluate(res['particles'], data['true_params'], 'VA')
        rm = Evaluator.compute_spectral_rmse(res['particles'], physics, data, seed=seed)
        print(f"{label:<22} {ev['thickness_rel_err']:<12.3f} {ev['porosity_rel_err']:<14.3f} "
              f"{rm['rmse_0']:<12.2e} {ev['avg_std']:<12.3f} {ev['f_std']:<10.4f}")


def sensitivity_starting_values(sigma: float = 1e-4, seed: int = 2024, *,
                                baseline_trial=None, porosity_only=False, **cheap):
    """改变 ASMC 提议中心和 VA 初始中心，保持先验和观测相同。"""
    grid = [('Design baseline', 1.0, None)]
    if not porosity_only:
        grid.extend([('Thickness -10%', 0.9, None), ('Thickness +10%', 1.1, None)])
    grid.append(('Porosity 0.40', 1.0, 0.40))
    print("\n" + "=" * 96)
    print(f"R2-7 起始值敏感性 (sigma={sigma:g}, seed={seed})")
    print("=" * 96)
    header = (f"{'起始配置':<18} {'方法':<12} {'厚度误差%':<12} {'孔隙率均值':<12} "
              f"{'孔隙率误差%':<14} {'RMSE 0°':<12} {'RMSE 30°':<12}")
    trials = {}
    for label, thickness_scale, initial_porosity in grid:
        if label == 'Design baseline' and baseline_trial is not None:
            print(f"\n复用主实验的 {label} 结果")
            trial = baseline_trial
        else:
            print(f"\n正在计算 {label} ...", flush=True)
            settings = {**cheap, 'INIT_D_SCALE': thickness_scale, 'INIT_F_VALUE': initial_porosity}
            trial = run_single_trial(sigma, seed, **settings)
        trials[label] = trial
        print(header)
        for method, name in (('smc', 'ASMC'), ('va', 'Gaussian VA')):
            e = trial[method]
            print(f"{label:<18} {name:<12} {e['thickness_rel_err']:<12.3f} "
                  f"{e['f_mean']:<12.4f} {e['porosity_rel_err']:<14.3f} "
                  f"{e['rmse_0']:<12.2e} {e['rmse_30']:<12.2e}")

    figure_path = ManuscriptPlotter().plot_starting_values(trials, sigma, seed)
    print(f"\n起始值对照图已保存: {figure_path}")


def compare_with_reference(sigma: float = 1e-4, seed: int = 2024, **cheap):
    """R2-8: 两个近似方法 vs 独立 Metropolis 参照解。"""
    config = Config()
    config.SIGMA_OBS = sigma
    for k, v in cheap.items():
        setattr(config, k, v)
    physics = PhysicsModel(config)
    data = DataGenerator(config, physics).generate(seed)
    true_d, true_dn, true_f = data['true_params']

    res_smc = SMCMethod(config, physics).run(data, seed)
    res_va = GaussianVA(config).to(torch.device(config.DEVICE)).run(data, physics, seed)
    res_ref = ReferenceMCMC(config, physics).run(data, seed)

    print("\n" + "=" * 96)
    print(f"R2-8 与独立参照解对比 (sigma={sigma:g}, seed={seed})")
    print("=" * 96)
    print(f"{'方法':<18} {'厚度误差%':<12} {'孔隙率误差%':<14} {'平均std(nm)':<12} {'f_std':<10}")
    print("-" * 66)
    for name, res in [('ASMC', res_smc), ('Gaussian VA', res_va), ('参照 MCMC', res_ref)]:
        ev = Evaluator.evaluate(res['particles'], data['true_params'], name)
        print(f"{name:<18} {ev['thickness_rel_err']:<12.3f} {ev['porosity_rel_err']:<14.3f} "
              f"{ev['avg_std']:<12.3f} {ev['f_std']:<10.4f}")
    print(f"\n  参照链接受率: {res_ref['acceptance_rate']:.3f}, "
          f"样本数: {len(res_ref['particles'])}")
    return {'smc': res_smc, 'va': res_va, 'ref': res_ref}


def compare_covariance_family(sigma: float = 1e-4, seed: int = 2024, **cheap):
    """R1-4: 对角 vs 全协方差 Gaussian VA 对比。"""
    config = Config()
    config.SIGMA_OBS = sigma
    for k, v in cheap.items():
        setattr(config, k, v)
    physics = PhysicsModel(config)
    data = DataGenerator(config, physics).generate(seed)

    va_diag = GaussianVA(config).to(torch.device(config.DEVICE))
    res_diag = va_diag.run(data, physics, seed)
    va_full = GaussianVAFullCov(config).to(torch.device(config.DEVICE))
    res_full = va_full.run(data, physics, seed)

    print("\n" + "=" * 96)
    print(f"R1-4 对角 vs 全协方差 Gaussian VA (sigma={sigma:g}, seed={seed})")
    print("=" * 96)
    for name, res in [('对角族', res_diag), ('全协方差族', res_full)]:
        ev = Evaluator.evaluate(res['particles'], data['true_params'], name)
        print(f"  {name:<10} 厚度误差 {ev['thickness_rel_err']:.3f}% | 孔隙率误差 "
              f"{ev['porosity_rel_err']:.3f}% | 平均std {ev['avg_std']:.3f} nm")
    corr = analyze_posterior_correlations(res_full['particles'])
    print("  全协方差族学到的最强参数关联（见上）。")
    return {'diag': res_diag, 'full': res_full, 'corr': corr}


# ============================================================
# Manuscript visualization routines
# ============================================================

class ManuscriptPlotter:
    """
    投稿版绘图引擎。
    """

    def __init__(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        # 1. 设定出版级绘图参数
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.linewidth': 1.0,
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'axes.labelpad': 8,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.frameon': False,
            'legend.fontsize': 9,
            'figure.dpi': 150,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'mathtext.fontset': 'stixsans',
        })

        # 2. 定义投稿图配色
        self.colors = {
            'obs': '#263238',  # 墨黑
            'gt': '#D50000',  # 绯红
            'smc': '#00838F',  # 深青 (ASMC)
            'smc_l': '#B2EBF2',  # 浅青
            'pinn': '#BF360C',  # 深赭 (Gaussian VA)
            'pinn_l': '#FFCCBC',  # 浅赭
            'grid': '#ECEFF1',  # 蓝灰
            'text': '#37474F'  # 深蓝灰
        }

        self.all_metrics = []

    def _style_axis(self, ax, grid=True):
        """去除边框，美化坐标轴"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['text'])
        ax.tick_params(axis='both', colors=self.colors['text'])
        if grid:
            ax.grid(True, linestyle='--', alpha=0.4, color=self.colors['grid'], zorder=0)

    def _add_panel_label(self, ax, label):
        ax.text(-0.05, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left', color='black')

    def plot_spectral_ab_test(self, data, res_smc, res_pinn, physics, sigma):
        """
        图1: 光谱拟合深度分析 (Double-Decker Layout with Zoom-Lens)
        R2-minor-3: 放大插图补数值刻度；置信带图例明确说明只含模型后验
        散布、不含观测噪声。
        """
        wl = data['wavelengths'].cpu().numpy()
        R_obs = data['R_obs'][0].cpu().numpy()

        def get_pred(res):
            # 可复现性: 子采样使用固定种子
            rng = np.random.default_rng(2024)
            idx = rng.choice(len(res['particles']), min(300, len(res['particles'])), replace=False)
            ps = torch.tensor(res['particles'][idx], device=physics.device, dtype=torch.float64)
            with torch.no_grad():
                R = physics.tmm_reflectance(ps[:, :5], ps[:, 5:10], ps[:, 10:], data['wavelengths'], 0.0)
            return R.mean(0).cpu().numpy(), R.std(0).cpu().numpy()

        mu_smc, std_smc = get_pred(res_smc)
        mu_pinn, std_pinn = get_pred(res_pinn)

        band_label = r'$\pm 2\sigma$ band (posterior spread, excl. obs. noise)'

        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1], hspace=0.1, wspace=0.15)

        # === ASMC panel ===
        ax_main_l = fig.add_subplot(gs[0, 0])
        ax_res_l = fig.add_subplot(gs[1, 0], sharex=ax_main_l)

        ax_main_l.plot(wl, mu_smc, color=self.colors['smc'], lw=2, label='Posterior Mean')
        ax_main_l.fill_between(wl, mu_smc - 2 * std_smc, mu_smc + 2 * std_smc, color=self.colors['smc_l'], alpha=0.6,
                               label=band_label)
        ax_main_l.scatter(wl[::4], R_obs[::4], color=self.colors['obs'], s=15, alpha=0.5, marker='o',
                          label='Observation')

        # ASMC zoom inset
        axins_l = inset_axes(ax_main_l, width="40%", height="35%", loc='lower left',
                             bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax_main_l.transAxes)
        zoom_range = (610, 650)
        zoom_mask = (wl > zoom_range[0]) & (wl < zoom_range[1])
        axins_l.plot(wl[zoom_mask], mu_smc[zoom_mask], color=self.colors['smc'], lw=2)
        axins_l.fill_between(wl[zoom_mask], (mu_smc - 2 * std_smc)[zoom_mask], (mu_smc + 2 * std_smc)[zoom_mask],
                             color=self.colors['smc_l'], alpha=0.6)
        axins_l.scatter(wl[zoom_mask], R_obs[zoom_mask], color=self.colors['obs'], s=20, alpha=0.6)
        # R2-minor-3: 插图显示数值刻度（原图无刻度）
        axins_l.tick_params(labelsize=7)
        axins_l.locator_params(nbins=4)
        for spine in axins_l.spines.values():
            spine.set_color(self.colors['smc'])
            spine.set_linewidth(1.5)
        mark_inset(ax_main_l, axins_l, loc1=2, loc2=4, fc="none", ec="0.5", ls=':')

        res_smc = R_obs - mu_smc
        ax_res_l.plot(wl, res_smc, color=self.colors['smc'], lw=1)
        ax_res_l.fill_between(wl, -2 * std_smc, 2 * std_smc, color='gray', alpha=0.1)
        ax_res_l.axhline(0, color='black', lw=0.8, ls='--')

        self._style_axis(ax_main_l)
        self._style_axis(ax_res_l)
        ax_main_l.set_xticklabels([])
        ax_main_l.set_ylabel('Reflectance ($R$)', fontsize=11)
        ax_res_l.set_ylabel('Residual ($\\Delta R$)', fontsize=11)
        ax_res_l.set_xlabel('Wavelength (nm)', fontsize=11)
        ax_main_l.set_title("ASMC Inversion", color=self.colors['smc'], fontsize=12, pad=10)
        self._add_panel_label(ax_main_l, '(a)')

        # === Gaussian VA panel ===
        ax_main_r = fig.add_subplot(gs[0, 1], sharey=ax_main_l)
        ax_res_r = fig.add_subplot(gs[1, 1], sharex=ax_main_r, sharey=ax_res_l)

        ax_main_r.plot(wl, mu_pinn, color=self.colors['pinn'], lw=2, label='Posterior Mean')
        ax_main_r.fill_between(wl, mu_pinn - 2 * std_pinn, mu_pinn + 2 * std_pinn, color=self.colors['pinn_l'],
                               alpha=0.6, label=band_label)
        ax_main_r.scatter(wl[::4], R_obs[::4], color=self.colors['obs'], s=15, alpha=0.5, marker='o')

        # Gaussian VA zoom inset
        axins_r = inset_axes(ax_main_r, width="40%", height="35%", loc='lower left',
                             bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax_main_r.transAxes)
        axins_r.plot(wl[zoom_mask], mu_pinn[zoom_mask], color=self.colors['pinn'], lw=2)
        axins_r.fill_between(wl[zoom_mask], (mu_pinn - 2 * std_pinn)[zoom_mask], (mu_pinn + 2 * std_pinn)[zoom_mask],
                             color=self.colors['pinn_l'], alpha=0.6)
        axins_r.scatter(wl[zoom_mask], R_obs[zoom_mask], color=self.colors['obs'], s=20, alpha=0.6)
        # R2-minor-3: 插图显示数值刻度
        axins_r.tick_params(labelsize=7)
        axins_r.locator_params(nbins=4)
        for spine in axins_r.spines.values():
            spine.set_color(self.colors['pinn'])
            spine.set_linewidth(1.5)
        mark_inset(ax_main_r, axins_r, loc1=2, loc2=4, fc="none", ec="0.5", ls=':')

        res_pinn = R_obs - mu_pinn
        ax_res_r.plot(wl, res_pinn, color=self.colors['pinn'], lw=1)
        ax_res_r.fill_between(wl, -2 * std_pinn, 2 * std_pinn, color='gray', alpha=0.1)
        ax_res_r.axhline(0, color='black', lw=0.8, ls='--')

        self._style_axis(ax_main_r)
        self._style_axis(ax_res_r)
        ax_main_r.set_xticklabels([])
        ax_main_r.set_yticklabels([])
        ax_res_r.set_yticklabels([])
        ax_res_r.set_xlabel('Wavelength (nm)', fontsize=11)
        ax_main_r.set_title("Gaussian VA Inversion", color=self.colors['pinn'], fontsize=12, pad=10)
        self._add_panel_label(ax_main_r, '(b)')

        handles, labels = ax_main_l.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=10)

        plt.suptitle(f'Spectral Reconstruction Analysis (Noise Level $\\sigma={sigma}$)', y=1.02, fontsize=14,
                     fontweight='bold')
        plt.savefig(OUTPUT_DIR / f'Fig1_Spectral_Analysis_Sigma_{sigma}_clean.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

    def plot_joint_distribution(self, res_smc, res_pinn, true_params, sigma):
        """
        图2: Joint-Distribution 联合分布拓扑图 (KDE + Marginals)
        """
        d1_smc, f_smc = res_smc['particles'][:, 0], res_smc['particles'][:, 10]
        d1_pinn, f_pinn = res_pinn['particles'][:, 0], res_pinn['particles'][:, 10]
        d1_true, f_true = true_params[0][0].item(), true_params[2].item()

        fig = plt.figure(figsize=(12, 6))
        outer_grid = gridspec.GridSpec(1, 2, wspace=0.25)

        def create_joint_plot(sub_gs, x, y, color, title, label_prefix):
            inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=sub_gs, wspace=0.05, hspace=0.05)
            ax_main = fig.add_subplot(inner_grid[1:, :-1])
            ax_top = fig.add_subplot(inner_grid[0, :-1], sharex=ax_main)
            ax_right = fig.add_subplot(inner_grid[1:, -1], sharey=ax_main)

            sns.kdeplot(x=x, y=y, ax=ax_main, color=color, fill=True, alpha=0.3, levels=6, thresh=0.05)
            sns.kdeplot(x=x, y=y, ax=ax_main, color=color, linewidths=1.5, levels=6, thresh=0.05)
            ax_main.scatter(d1_true, f_true, color=self.colors['gt'], marker='*', s=200, zorder=10, edgecolors='white',
                            linewidth=1.2, label='Truth')
            ax_main.scatter(x.mean(), y.mean(), color='white', marker='P', s=80, zorder=9, edgecolors='black',
                            linewidth=1.2, label='Mean')

            ax_top.hist(x, bins=30, density=True, color=color, alpha=0.6, edgecolor='none')
            ax_top.axvline(d1_true, color=self.colors['gt'], linestyle='--', linewidth=1)
            ax_right.hist(y, bins=30, density=True, orientation='horizontal', color=color, alpha=0.6, edgecolor='none')
            ax_right.axhline(f_true, color=self.colors['gt'], linestyle='--', linewidth=1)

            self._style_axis(ax_main)
            ax_top.axis('off')
            ax_right.axis('off')
            ax_main.set_xlabel(r'Thickness $d_{SiO2}$ (nm)', fontsize=10)
            ax_main.set_ylabel(r'Porosity $f_{ULD}$', fontsize=10)
            ax_top.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)
            return ax_main

        ax_smc = create_joint_plot(outer_grid[0], d1_smc, f_smc, self.colors['smc'], "ASMC Joint Posterior", "(c)")
        self._add_panel_label(fig.axes[0], '(c)')

        ax_pinn = create_joint_plot(outer_grid[1], d1_pinn, f_pinn, self.colors['pinn'], "Gaussian VA Joint Posterior", "(d)")
        self._add_panel_label(fig.axes[3], '(d)')

        handles, labels = ax_smc.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=10)

        plt.suptitle(f'Parameter Coupling & Joint Distribution (Noise $\\sigma={sigma}$)', y=0.98, fontsize=14,
                     fontweight='bold')
        plt.savefig(OUTPUT_DIR / f'Fig2_Joint_Topology_Sigma_{sigma}_clean.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

    def add_metrics(self, metrics):
        self.all_metrics.append(metrics)

    def plot_starting_values(self, trials, sigma, seed):
        """Plot paired reconstruction errors for each initialization."""
        labels = list(trials)
        positions = np.arange(len(labels))
        width = 0.35
        panels = (
            ('thickness_rel_err', 'Mean thickness error (%)'),
            ('porosity_rel_err', 'Porosity error (%)'),
            ('rmse_0', r'Spectral RMSE at $0^\circ$'),
            ('rmse_30', r'Spectral RMSE at $30^\circ$'),
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for ax, (metric, title) in zip(axes.flat, panels):
            self._style_axis(ax)
            ax.bar(positions - width / 2, [trial['smc'][metric] for trial in trials.values()],
                   width, color=self.colors['smc'], label='ASMC')
            ax.bar(positions + width / 2, [trial['va'][metric] for trial in trials.values()],
                   width, color=self.colors['pinn'], label='Gaussian VA')
            ax.set_xticks(positions, labels)
            ax.set_title(title, loc='left')
            if metric.startswith('rmse'):
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axes[0, 0].legend()
        fig.suptitle(f'Initialization sensitivity ($\\sigma={sigma:g}$, seed {seed})')
        path = OUTPUT_DIR / f'Fig_Starting_Values_Sigma_{sigma:g}_Seed_{seed}.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_slopegraph_summary(self):
        """Plot the mean across seeds at each noise level."""
        metrics_by_noise = {}
        for metrics in self.all_metrics:
            metrics_by_noise.setdefault(metrics['sigma'], []).append(metrics)
        noise_levels = sorted(metrics_by_noise)
        if not noise_levels:
            return

        metric_groups = [metrics_by_noise[sigma] for sigma in noise_levels]
        noise_labels = [f'$\\sigma={sigma:g}$' for sigma in noise_levels]
        x_pos = range(len(noise_levels))
        panels = (
            ('spectral_rmse', 'Spectral RMSE (0°)', 'RMSE (log scale)', True),
            ('thickness_rel_err', 'Thickness Error', 'Relative Error (%)', False),
            ('porosity_rel_err', 'Porosity Error', 'Relative Error (%)', False),
            ('runtime', 'Computing Time', 'Seconds', False),
        )

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

        for idx, (metric, title, unit, is_log) in enumerate(panels):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            y_smc = [np.mean([row[f'{metric}_smc'] for row in group]) for group in metric_groups]
            y_pinn = [np.mean([row[f'{metric}_pinn'] for row in group]) for group in metric_groups]

            ax.plot(x_pos, y_smc, '-', color=self.colors['smc'], lw=2, alpha=0.8, label='ASMC' if idx == 0 else "")
            ax.plot(x_pos, y_pinn, '-', color=self.colors['pinn'], lw=2, alpha=0.8, label='Gaussian VA' if idx == 0 else "")

            ax.scatter(x_pos, y_smc, c=self.colors['smc'], s=100, zorder=5, edgecolors='white', lw=1.5)
            ax.scatter(x_pos, y_pinn, c=self.colors['pinn'], s=100, zorder=5, marker='D', edgecolors='white', lw=1.5)

            for i, vs, vp in zip(x_pos, y_smc, y_pinn):
                offset_scale = 0.6 if is_log else 1.0
                s_top = vs >= vp
                offset_s = (12 if s_top else -20) * offset_scale
                offset_p = (-20 if s_top else 12) * offset_scale
                fmt = '{:.2e}' if is_log else '{:.3g}'
                ax.annotate(fmt.format(vs), (i, vs), xytext=(0, offset_s), textcoords='offset points',
                            ha='center', color=self.colors['smc'], fontweight='bold', fontsize=9)
                ax.annotate(fmt.format(vp), (i, vp), xytext=(0, offset_p), textcoords='offset points',
                            ha='center', color=self.colors['pinn'], fontweight='bold', fontsize=9)

            self._style_axis(ax, grid=True)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(noise_labels, fontweight='bold')
            ax.set_xlim(-0.2, x_pos[-1] + 0.2)

            if is_log:
                ax.set_yscale('log')
                ymin = min(min(y_smc), min(y_pinn)) * 0.4
                ymax = max(max(y_smc), max(y_pinn)) * 6.0
                ax.set_ylim(ymin, ymax)
            else:
                ymin = min(min(y_smc), min(y_pinn)) * 0.8
                ymax = max(max(y_smc), max(y_pinn)) * 1.2
                ax.set_ylim(ymin, ymax)

            ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
            ax.set_ylabel(unit, fontsize=10, color=self.colors['text'])
            self._add_panel_label(ax, f"({chr(97 + idx + 4)})")

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=self.colors['smc'], lw=2, marker='o'),
                        Line2D([0], [0], color=self.colors['pinn'], lw=2, marker='D')]
        fig.legend(custom_lines, ['ASMC', 'Gaussian VA'],
                   loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=11)

        fig.suptitle('Noise-level comparison (mean across seeds)', fontsize=16, fontweight='bold', y=0.99)
        plt.savefig(OUTPUT_DIR / 'Supplementary_Performance_Trends.png', bbox_inches='tight', dpi=300)
        plt.close(fig)


# ============================================================
# 主执行流程
# ============================================================

def run_noise_level_comparison(noise_levels: List[float] = [1e-4, 5e-3],
                               seeds: List[int] = [2024, 2025, 2026]):
    """R1-5: 每个噪声水平在多个独立种子下重复，报告均值±标准差。

    光谱和联合分布图使用每档噪声的最后一个种子。汇总图使用各档噪声的多种子均值。
    最高噪声档的第一个种子做完整初始化对照，其余种子只重复孔隙率偏置。
    """
    print("=" * 90)
    print(f"Generating manuscript figures and detailed console output (seeds={seeds})")
    print("=" * 90)

    validate_forward_model()
    print_prior_specification()

    plotter = ManuscriptPlotter()
    evaluator = Evaluator()
    trials = []

    for sigma in noise_levels:
        print(f"\n>>> Processing Noise Level σ = {sigma} ...")
        config = Config()
        config.SIGMA_OBS = sigma
        physics = PhysicsModel(config)

        for seed in seeds:
            print(f"   [seed = {seed}]")
            data_gen = DataGenerator(config, physics)
            data = data_gen.generate(seed)

            print("   Running ASMC...")
            smc = SMCMethod(config, physics)
            res_smc = smc.run(data, seed)
            runtime_smc = res_smc['runtime']

            print("   Running Gaussian VA...")
            pinn = GaussianVA(config).to(torch.device(config.DEVICE))
            res_pinn = pinn.run(data, physics, seed)
            runtime_pinn = res_pinn['runtime']

            metrics = evaluator.print_detailed_metrics(sigma, res_smc, res_pinn, data,
                                                       runtime_smc, runtime_pinn,
                                                       physics=physics, seed=seed)
            trials.append({'sigma': sigma, 'seed': seed,
                           'smc': {**Evaluator.evaluate(res_smc['particles'], data['true_params'], 'ASMC'),
                                   'rmse_0': metrics['spectral_rmse_smc'],
                                   'rmse_30': metrics['spectral_rmse30_smc'],
                                   'runtime': runtime_smc},
                           'va': {**Evaluator.evaluate(res_pinn['particles'], data['true_params'], 'Gaussian VA'),
                                  'rmse_0': metrics['spectral_rmse_pinn'],
                                  'rmse_30': metrics['spectral_rmse30_pinn'],
                                  'runtime': runtime_pinn}})
            plotter.add_metrics(metrics)

            last_data, last_res_smc, last_res_pinn = data, res_smc, res_pinn

        print("   Rendering high-resolution figures...")
        plotter.plot_spectral_ab_test(last_data, last_res_smc, last_res_pinn, physics, sigma)
        plotter.plot_joint_distribution(last_res_smc, last_res_pinn, last_data['true_params'], sigma)
        analyze_posterior_correlations(last_res_smc['particles'])
        analyze_posterior_correlations(last_res_pinn['particles'])

    print_seed_statistics(trials)
    print("   Rendering summary slopegraph...")
    plotter.plot_slopegraph_summary()
    if trials:
        highest_noise = max(trial['sigma'] for trial in trials)
        high_noise_trials = [trial for trial in trials if trial['sigma'] == highest_noise]
        for index, baseline in enumerate(high_noise_trials):
            sensitivity_starting_values(highest_noise, baseline['seed'], baseline_trial=baseline,
                                        porosity_only=(index > 0))
    print("\nAll figures generated successfully.")


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="ASMC vs Gaussian VA 对比实验。直接运行时执行论文主实验及高噪声初值对照。")
    parser.add_argument('--mode', default='paper',
                        choices=['paper', 'validate', 'priors', 'porosity',
                                 'asmc-sensitivity', 'va-sensitivity',
                                 'starting-values', 'reference', 'full-cov'],
                        help="默认 'paper'。'starting-values' 单独运行初值对照。")
    parser.add_argument('--sigma', type=float, nargs='+',
                        help="初值对照默认 0.005，其他模式沿用 0.0001 和 0.005。")
    parser.add_argument('--seed', type=int, nargs='+', default=[2024, 2025, 2026])
    parser.add_argument('--particles', type=int, default=None, help="覆盖 N_PARTICLES_SMC")
    parser.add_argument('--stages', type=int, default=None, help="覆盖 T_STAGES_SMC")
    parser.add_argument('--epochs', type=int, default=None, help="覆盖 EPOCHS_VA")
    parser.add_argument('--mcmc-steps', type=int, default=None, help="覆盖 REF_MCMC_STEPS")
    return parser


if __name__ == "__main__":
    args = _build_argument_parser().parse_args()
    if args.sigma is None:
        args.sigma = [5e-3] if args.mode == 'starting-values' else [1e-4, 5e-3]

    cheap = {}
    if args.particles is not None:
        cheap['N_PARTICLES_SMC'] = args.particles
    if args.stages is not None:
        cheap['T_STAGES_SMC'] = args.stages
    if args.epochs is not None:
        cheap['EPOCHS_VA'] = args.epochs
    if args.mcmc_steps is not None:
        cheap['REF_MCMC_STEPS'] = args.mcmc_steps

    if args.mode == 'paper':
        run_noise_level_comparison(noise_levels=list(args.sigma), seeds=list(args.seed))
    elif args.mode == 'validate':
        validate_forward_model(sigma=args.sigma[0])
    elif args.mode == 'priors':
        print_prior_specification()
    elif args.mode == 'porosity':
        porosity_forward_sensitivity(sigma=args.sigma[0], seed=args.seed[0])
    elif args.mode == 'asmc-sensitivity':
        sensitivity_asmc_settings(sigma=args.sigma[-1], seed=args.seed[0], **cheap)
    elif args.mode == 'va-sensitivity':
        sensitivity_va_settings(sigma=args.sigma[-1], seed=args.seed[0], **cheap)
    elif args.mode == 'starting-values':
        sensitivity_starting_values(sigma=args.sigma[0], seed=args.seed[0], **cheap)
    elif args.mode == 'reference':
        compare_with_reference(sigma=args.sigma[0], seed=args.seed[0], **cheap)
    elif args.mode == 'full-cov':
        compare_covariance_family(sigma=args.sigma[0], seed=args.seed[0], **cheap)
