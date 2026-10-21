"""
光学薄膜反演方法对比实验：ASMC vs Gaussian VA
投稿版可视化与数值记录 (Joint-KDE + Zoom-in Insets)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns
from scipy.stats import gaussian_kde
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

warnings.filterwarnings('ignore')
OUTPUT_DIR = Path(__file__).resolve().parent / "figs"

# R1-4：参数名，用于后验相关性报告
LAYER_NAMES = ['d1_ULD', 'd2_HfO2', 'd3_SiO2', 'd4_Ta2O5', 'd5_MgF2',
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

    # ---- R1-1：似然加权与折射率约束的可切换开关 ----
    # 默认值全部保持原实现行为，因此默认运行结果与修改前完全一致；
    # 关闭开关后重跑即可量化这些"特定选择"对 Table 3 数值的影响。
    USE_WEIGHTED_LIKELIHOOD: bool = True    # True = 设计波长 632.8nm 处十倍加权（原实现）
    DESIGN_WAVELENGTH_WEIGHT: float = 10.0  # 加权倍数（仅 USE_WEIGHTED_LIKELIHOOD=True 时生效）
    ENABLE_INDEX_CLAMPS: bool = True        # True = 启用 HfO2/Ta2O5 及多孔层表面折射率钳位（原实现）
    # True = Gaussian VA 与 ASMC 共用同一个加权高斯似然（逐样本 Monte-Carlo ELBO）。
    # 原实现中两法目标并不相同：ASMC 用逐粒子加权似然，VA 用"先对预测谱取平均、
    # 再算无权重 MSE"。审稿人 1 要求澄清并统一两法的加权目标，故默认置 True，
    # 两法共用一致的波长加权与逐样本似然。
    # 注意一：置 True 会改变 Table 3 全部数值与全部图表，必须重跑重绘。
    # 注意二：置 False 可回到原有（两法目标不同）的结果，用于前后对照。
    # 注意三：VA 的数据项保留似然中的 1/sigma^2（不再做 * SIGMA_OBS**2 抵消），
    #         故 VA 与 ASMC 使用相同的噪声假设，后验宽度随噪声水平变化。
    #         数据项改为 per-datum 尺度，量级随 sigma 变化，必要时调整 KL_WEIGHT。
    VA_SHARED_LIKELIHOOD: bool = True

    # 计算设备
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ASMC参数
    N_WAVELENGTHS: int = 120
    N_PARTICLES_SMC: int = 4000
    T_STAGES_SMC: int = 50
    ESS_THRESHOLD_FRAC: float = 0.5   # R1-3: 重采样阈值（原实现硬编码 0.5N）
    JITTER_SCALE: float = 0.5         # R1-3: 协方差抖动尺度（原实现硬编码 0.5）

    # Gaussian VA hyperparameters
    EPOCHS_VA: int = 5000
    LR_VA: float = 5e-4
    N_SAMPLES_VA: int = 16
    KL_WEIGHT: float = 1e-6

    def __post_init__(self):
        if self.MATERIALS is None:
            self.MATERIALS = ['SiO2-ULD', 'HfO2', 'SiO2', 'Ta2O5', 'MgF2']


# ============================================================
# 物理模型
# ============================================================

class PhysicsModel:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

    def sellmeier_sio2(self, wl_nm):
        wl = wl_nm * 1e-3
        B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
        C1, C2, C3 = 0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2
        n2 = 1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2) + B3 * wl ** 2 / (wl ** 2 - C3)
        return torch.sqrt(torch.clamp(n2, min=1.45 ** 2))

    def sellmeier_hfo2(self, wl_nm):
        # R1-1: 钳位由 ENABLE_INDEX_CLAMPS 控制，默认 True 与原实现一致
        wl = wl_nm * 1e-3
        B1, B2, B3 = 1.50, 0.55, 1.20
        C1, C2, C3 = 0.05 ** 2, 0.10 ** 2, 8.00 ** 2
        n2 = 1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2) + B3 * wl ** 2 / (wl ** 2 - C3)
        n = torch.sqrt(n2)
        if self.config.ENABLE_INDEX_CLAMPS:
            n = torch.clamp(n, min=1.94, max=1.96)
        return n

    def sellmeier_ta2o5(self, wl_nm):
        # R1-1: 同上，默认 True 与原实现一致
        wl = wl_nm * 1e-3
        B1, B2, B3 = 2.13, 0.58, 1.54
        C1, C2, C3 = 0.062 ** 2, 0.16 ** 2, 13.73 ** 2
        n2 = 1 + B1 * wl ** 2 / (wl ** 2 - C1) + B2 * wl ** 2 / (wl ** 2 - C2) + B3 * wl ** 2 / (wl ** 2 - C3)
        n = torch.sqrt(n2)
        if self.config.ENABLE_INDEX_CLAMPS:
            n = torch.clamp(n, min=2.08, max=2.12)
        return n

    def sellmeier_mgf2(self, wl_nm):
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
        if grad_strength is None:
            grad_strength = self.config.MOTH_EYE_GRAD_STRENGTH
        f_sio2 = 1 - f
        eps_m = self.config.AIR_N ** 2
        eps_i = n_uld ** 2
        eps_eff = eps_m * ((eps_i + 2 * eps_m + 2 * f_sio2 * (eps_i - eps_m)) /
                           (eps_i + 2 * eps_m - f_sio2 * (eps_i - eps_m)))
        n_base = torch.sqrt(torch.clamp(eps_eff.real, min=1e-6))
        n_surface = 1.09 + (n_base - 1.09) * (1 - grad_strength ** 2.2)
        # R1-1 / R1-6: 表面折射率钳位可关闭（默认 True 与原实现一致）。
        # 该钳位在 f 的先验区间内会饱和，是孔隙率前向敏感度偏低的原因之一，
        # 置 False 后重跑即可对比关闭前后的差别。
        if self.config.ENABLE_INDEX_CLAMPS:
            n_surface = torch.clamp(n_surface, min=1.09, max=1.10)
        return n_surface

    def tmm_reflectance(self, d, dn, f, wl, theta_deg):
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
        theta = torch.deg2rad(torch.tensor(theta_deg, device=self.device))
        sin_theta0 = torch.sin(theta)

        for i in range(5):
            ni = n_layers[..., i]
            di = d[:, i].view(-1, 1)
            sin_thetai = (self.config.AIR_N / ni) * sin_theta0
            sin_thetai = torch.clamp(sin_thetai.real, -0.99999, 0.99999)
            cos_thetai = torch.sqrt(1 - sin_thetai ** 2).to(dtype_c)
            delta = k0 * ni * di * cos_thetai
            cos_d = torch.cos(delta)
            sin_d = torch.sin(delta)
            m11 = cos_d
            m12 = 1j * sin_d / (ni * cos_thetai)
            m21 = 1j * ni * cos_thetai * sin_d
            m22 = cos_d
            Mi = torch.stack([torch.stack([m11, m12], dim=-1), torch.stack([m21, m22], dim=-1)], dim=-2)
            M = torch.matmul(M, Mi)

        m11, m12 = M[..., 0, 0], M[..., 0, 1]
        m21, m22 = M[..., 1, 0], M[..., 1, 1]
        ns = self.config.SUBSTRATE_N
        num = m11 + m12 * ns - m21 / ns - m22
        den = m11 + m12 * ns + m21 / ns + m22
        r = num / (den + 1e-10)
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

        noise_0 = self.config.SIGMA_OBS * torch.randn_like(R_true_0)
        noise_30 = self.config.SIGMA_OBS * torch.randn_like(R_true_30)
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

def weighted_gaussian_loglik(d, dn, f, wl, R_obs_0, R_obs_30, physics, config, no_grad=True):
    """R1-1：ASMC 与 Gaussian VA 共用的同一个加权高斯对数似然。

    审稿人 1 要求"说明该加权的统计/物理依据，并澄清两种方法是否使用完全相同的
    加权目标"。原实现里两法目标并不一致：ASMC 走 SMCMethod.log_likelihood_batch
    （逐粒子、632.8 nm 处十倍加权、带 1/sigma^2），Gaussian VA 走的是"先对预测
    谱取样本平均、再算无权重 MSE"，既无波长加权也无 1/sigma^2。本函数把该目标
    抽成唯一实现，两法共用，从而：

      * USE_WEIGHTED_LIKELIHOOD=True  -> 632.8 nm 处 DESIGN_WAVELENGTH_WEIGHT 倍加权；
      * USE_WEIGHTED_LIKELIHOOD=False -> 等权，退化为标准高斯观测模型。

    权重按 sum(w)=1 归一化，因此加权只改变各波长点的相对贡献，不改变似然的整体
    量级，便于与等权情形直接比较。

    no_grad=True  等价于原 ASMC 行为（不需要梯度，省去建图开销）；
    no_grad=False 保留计算图，供 Gaussian VA 的 Monte-Carlo ELBO 反传。
    """
    if no_grad:
        with torch.no_grad():
            R_pred_0 = physics.tmm_reflectance(d, dn, f, wl, 0.0)
            R_pred_30 = physics.tmm_reflectance(d, dn, f, wl, 30.0)
    else:
        R_pred_0 = physics.tmm_reflectance(d, dn, f, wl, 0.0)
        R_pred_30 = physics.tmm_reflectance(d, dn, f, wl, 30.0)

    weights = torch.ones_like(wl)
    if config.USE_WEIGHTED_LIKELIHOOD:
        center_idx = torch.argmin(torch.abs(wl - config.DESIGN_WAVELENGTH))
        weights[center_idx] = weights[center_idx] * float(config.DESIGN_WAVELENGTH_WEIGHT)
    weights = weights / weights.sum()
    err_0 = torch.sum(weights * (R_obs_0 - R_pred_0) ** 2, dim=-1)
    err_30 = torch.sum(weights * (R_obs_30 - R_pred_30) ** 2, dim=-1)
    ll_0 = -0.5 * err_0 / (config.SIGMA_OBS ** 2)
    ll_30 = -0.5 * err_30 / (config.SIGMA_OBS ** 2)
    return ll_0 + ll_30


class SMCMethod:
    def __init__(self, config: Config, physics: PhysicsModel):
        self.config = config
        self.physics = physics
        self.device = torch.device(config.DEVICE)

    def log_likelihood_batch(self, particles, R_obs_0, R_obs_30, wl):
        particles_tensor = torch.tensor(particles, device=self.device, dtype=torch.float64)
        d = particles_tensor[:, :5]
        dn = particles_tensor[:, 5:10]
        f = particles_tensor[:, 10:11]
        # R1-1: 与 Gaussian VA 共用同一个加权高斯似然（见 weighted_gaussian_loglik）。
        # 加权由 USE_WEIGHTED_LIKELIHOOD 控制：默认 True 与原实现逐位一致；
        # 置 False 则退化为等权（标准高斯似然），用于量化该加权的影响。
        return weighted_gaussian_loglik(d, dn, f, wl, R_obs_0, R_obs_30,
                                        self.physics, self.config, no_grad=True).cpu().numpy()

    def run(self, data: Dict, seed: int = 42) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        R_obs_0, R_obs_30 = data['R_obs']
        wl = data['wavelengths']
        N = self.config.N_PARTICLES_SMC
        T = self.config.T_STAGES_SMC
        d_design = torch.tensor([
            self.config.DESIGN_WAVELENGTH / (4 * 1.45), self.config.DESIGN_WAVELENGTH / (4 * 1.95),
            self.config.DESIGN_WAVELENGTH / (4 * 1.45), self.config.DESIGN_WAVELENGTH / (4 * 2.05),
            self.config.DESIGN_WAVELENGTH / (4 * 1.38)
        ])
        particles = np.hstack([
            d_design.numpy() + 0.5 * np.random.randn(N, 5),
            np.random.randn(N, 5) * 0.002,
            np.random.randn(N, 1) * 0.005 + self.config.TRUE_F_ULD
        ])
        logw = np.zeros(N)
        betas = np.linspace(0, 1, T)
        ess_history = []
        start_time = time.time()

        for t in range(1, T):
            ll = self.log_likelihood_batch(particles, R_obs_0, R_obs_30, wl)
            logw += (betas[t] - betas[t - 1]) * ll
            logw -= np.max(logw)
            w = np.exp(logw)
            w /= np.sum(w)
            ess = 1.0 / np.sum(w ** 2)
            ess_history.append(ess)
            # R1-3: 阈值与抖动尺度改由 Config 提供（默认值 0.5 / 0.5 与原实现一致）
            if ess < N * self.config.ESS_THRESHOLD_FRAC:
                idx = np.random.choice(N, size=N, p=w)
                particles = particles[idx]
                logw[:] = 0.0
            cov = np.cov(particles.T) + 1e-8 * np.eye(11)
            particles += np.random.multivariate_normal(np.zeros(11), self.config.JITTER_SCALE * cov, size=N)
            particles[:, :5] = np.clip(particles[:, :5], 40, 150)
            particles[:, 5:10] = np.clip(particles[:, 5:10], -0.02, 0.02)
            particles[:, 10] = np.clip(particles[:, 10], 0.30, 0.50)
            if t % 10 == 0:
                print(f"  SMC进度: {t}/{T} | ESS: {ess:.0f}/{N}")
        return {'particles': particles, 'ess_history': ess_history, 'runtime': time.time() - start_time}


class GaussianVA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.DEVICE)
        d_init = torch.tensor([
            config.DESIGN_WAVELENGTH / (4 * 1.45), config.DESIGN_WAVELENGTH / (4 * 1.95),
            config.DESIGN_WAVELENGTH / (4 * 1.45), config.DESIGN_WAVELENGTH / (4 * 2.05),
            config.DESIGN_WAVELENGTH / (4 * 1.38)
        ], device=self.device)
        self.d_mu = nn.Parameter(d_init)
        self.d_log_std = nn.Parameter(torch.ones(5, device=self.device) * np.log(5.0))
        self.dn_mu = nn.Parameter(torch.zeros(5, device=self.device))
        self.dn_log_std = nn.Parameter(torch.ones(5, device=self.device) * np.log(0.003))
        self.f_mu = nn.Parameter(torch.tensor([config.TRUE_F_ULD], device=self.device))
        self.f_log_std = nn.Parameter(torch.ones(1, device=self.device) * np.log(0.02))
        self.d_bounds = (40.0, 150.0)
        self.dn_bounds = (-0.02, 0.02)
        self.f_bounds = (0.30, 0.50)

    def sample(self, n_samples=1):
        d = torch.clamp(Normal(self.d_mu, torch.exp(self.d_log_std)).rsample((n_samples,)), *self.d_bounds)
        dn = torch.clamp(Normal(self.dn_mu, torch.exp(self.dn_log_std)).rsample((n_samples,)), *self.dn_bounds)
        f = torch.clamp(Normal(self.f_mu, torch.exp(self.f_log_std)).rsample((n_samples,)), *self.f_bounds)
        return d, dn, f

    def kl_loss(self):
        kl_d = -0.5 * torch.sum(1 + 2 * self.d_log_std - self.d_mu ** 2 - torch.exp(2 * self.d_log_std))
        kl_dn = -0.5 * torch.sum(1 + 2 * self.dn_log_std - self.dn_mu ** 2 - torch.exp(2 * self.dn_log_std))
        kl_f = -0.5 * torch.sum(1 + 2 * self.f_log_std - self.f_mu ** 2 - torch.exp(2 * self.f_log_std))
        return kl_d + kl_dn + kl_f

    def run(self, data: Dict, physics: PhysicsModel, seed: int = 42) -> Dict:
        torch.manual_seed(seed)
        R_obs_0, R_obs_30 = data['R_obs']
        wl = data['wavelengths']
        optimizer = optim.Adam(self.parameters(), lr=self.config.LR_VA)
        loss_fn = nn.MSELoss()
        start_time = time.time()
        loss_history = []
        for epoch in range(self.config.EPOCHS_VA):
            optimizer.zero_grad()
            d_s, dn_s, f_s = self.sample(self.config.N_SAMPLES_VA)
            if self.config.VA_SHARED_LIKELIHOOD:
                # R1-1（噪声假设）：与 ASMC 使用完全相同的加权高斯似然，逐样本
                # Monte-Carlo ELBO，不再先对预测谱取平均。
                # 关键：保留似然中的 1/sigma^2，让噪声水平真正进入 VA 的目标——
                # 此前 (* SIGMA_OBS**2) 正好把它抵消，使 VA 的后验宽度与噪声水平
                # 无关（即 R2-8 所指"两噪声水平下散布几乎相同"的现象）。
                # 除以观测点数（2 个角度 × N_lambda）换算到 per-datum 尺度，
                # 与 MSELoss 的 reduction='mean' 同义。
                # 注意：数据项现在随 sigma 减小而增大，若后验过紧，需上调
                # Config.KL_WEIGHT 以恢复数据项与正则项的相对权重。
                ll = weighted_gaussian_loglik(d_s, dn_s, f_s, wl, R_obs_0, R_obs_30,
                                              physics, self.config, no_grad=False)
                loss_data = -ll.mean() / (2.0 * wl.numel())
            else:
                R_p0 = physics.tmm_reflectance(d_s, dn_s, f_s, wl, 0.0)
                R_p30 = physics.tmm_reflectance(d_s, dn_s, f_s, wl, 30.0)
                loss_data = loss_fn(torch.cat([R_p0.mean(0), R_p30.mean(0)]), torch.cat([R_obs_0, R_obs_30]))
            loss_kl = self.kl_loss() * self.config.KL_WEIGHT
            loss = loss_data + loss_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            loss_history.append(loss.item())
            if (epoch + 1) % 500 == 0:
                print(f"  Gaussian VA progress: {epoch + 1}/{self.config.EPOCHS_VA} | loss: {loss.item():.6f}")
        with torch.no_grad():
            d_post, dn_post, f_post = self.sample(4000)
        return {'particles': torch.cat([d_post, dn_post, f_post], dim=1).cpu().numpy(), 'loss_history': loss_history,
                'runtime': time.time() - start_time}


# ============================================================
# 评估器
# ============================================================

class Evaluator:
    @staticmethod
    def evaluate(particles: np.ndarray, true_params: Tuple, method_name: str) -> Dict:
        true_d, true_dn, true_f = [p.cpu().numpy() if torch.is_tensor(p) else p for p in true_params]
        d_mean = particles[:, :5].mean(axis=0)
        d_std = particles[:, :5].std(axis=0)
        d_err = np.abs(d_mean - true_d) / true_d * 100
        dn_std = particles[:, 5:10].std(axis=0)
        f_mean = particles[:, 10].mean()
        f_std = particles[:, 10].std()
        f_err = np.abs(f_mean - true_f) / true_f * 100

        # R1-2：经验 2-sigma 覆盖率——真值是否落在 [均值-2std, 均值+2std] 内。
        # 这是检验"Gaussian VA 的窄散布是否只是参数化产物"的直接指标：
        # 若散布标定正确，覆盖率应接近 0.954；显著偏低即为过度自信。
        d_covered = np.abs(d_mean - true_d) <= 2.0 * d_std
        f_covered = np.abs(f_mean - true_f) <= 2.0 * f_std

        return {
            'method': method_name,
            'thickness_rmse': np.sqrt(np.mean((d_mean - true_d) ** 2)),
            'thickness_mae': np.mean(np.abs(d_mean - true_d)),
            'thickness_rel_err': np.mean(d_err),
            'porosity_rel_err': f_err.item(),
            'avg_std': np.mean(d_std),
            'd_mean': d_mean.tolist(),
            'd_std': d_std.tolist(),
            'f_mean': float(f_mean),
            'f_std': float(f_std),
            # 以下为 R1-2 新增字段（不改动上面任何原有字段）
            'thickness_coverage_2sigma': float(np.mean(d_covered)),
            'porosity_coverage_2sigma': float(f_covered.item())
        }

    @staticmethod
    def print_detailed_metrics(sigma, res_smc, res_pinn, data, runtime_smc, runtime_pinn,
                               seed: int = 2024):
        eval_smc = Evaluator.evaluate(res_smc['particles'], data['true_params'], 'ASMC')
        eval_pinn = Evaluator.evaluate(res_pinn['particles'], data['true_params'], 'Gaussian VA')
        wl = data['wavelengths'].cpu().numpy()
        R_obs = data['R_obs'][0].cpu().numpy()

        def get_spectral_rmse(res, physics):
            # 用独立 RandomState 而非全局 np.random：原实现未设种子，
            # 同一份结果每次运行会给出不同的光谱 RMSE，无法复现。
            idx = np.random.RandomState(seed).choice(len(res['particles']), 200)
            ps = torch.tensor(res['particles'][idx], device=physics.device, dtype=torch.float32)
            with torch.no_grad():
                R = physics.tmm_reflectance(ps[:, :5], ps[:, 5:10], ps[:, 10:], data['wavelengths'], 0.0)
            mu = R.mean(0).cpu().numpy()
            return np.sqrt(np.mean((mu - R_obs) ** 2))

        physics = PhysicsModel(Config())
        spectral_rmse_smc = get_spectral_rmse(res_smc, physics)
        spectral_rmse_pinn = get_spectral_rmse(res_pinn, physics)

        print("\n" + "=" * 80)
        print(f"噪声水平 sigma={sigma} - 核心对比数据")
        print("=" * 80)
        print("\n【基础性能对比】")
        print(f"{'指标':<20} {'ASMC':<15} {'Gaussian VA':<15} {'优势方':<10}")
        print("-" * 60)
        print(
            f"{'光谱RMSE':<20} {spectral_rmse_smc:.2e} {spectral_rmse_pinn:.2e} {('ASMC' if spectral_rmse_smc < spectral_rmse_pinn else 'Gaussian VA'):<10}")
        print(
            f"{'厚度相对误差(%)':<20} {eval_smc['thickness_rel_err']:.2f} {eval_pinn['thickness_rel_err']:.2f} {('ASMC' if eval_smc['thickness_rel_err'] < eval_pinn['thickness_rel_err'] else 'Gaussian VA'):<10}")
        print(
            f"{'孔隙率相对误差(%)':<20} {eval_smc['porosity_rel_err']:.2f} {eval_pinn['porosity_rel_err']:.2f} {('ASMC' if eval_smc['porosity_rel_err'] < eval_pinn['porosity_rel_err'] else 'Gaussian VA'):<10}")
        print(
            f"{'运行时间(s)':<20} {runtime_smc:.2f} {runtime_pinn:.2f} {('ASMC' if runtime_smc < runtime_pinn else 'Gaussian VA'):<10}")

        print("\n【逐层厚度相对误差(%)】")
        layers = ['L1: SiO2-ULD', 'L2: HfO2', 'L3: SiO2', 'L4: Ta2O5', 'L5: MgF2']
        true_d = data['true_params'][0].cpu().numpy()
        d_mean_smc = res_smc['particles'][:, :5].mean(axis=0)
        d_mean_pinn = res_pinn['particles'][:, :5].mean(axis=0)
        err_smc = np.abs(d_mean_smc - true_d) / true_d * 100
        err_pinn = np.abs(d_mean_pinn - true_d) / true_d * 100
        print(f"{'层名':<15} {'ASMC误差(%)':<15} {'Gaussian VA误差(%)':<15}")
        print("-" * 45)
        for i, layer in enumerate(layers):
            print(f"{layer:<15} {err_smc[i]:<15.2f} {err_pinn[i]:<15.2f}")

        print("\n【后验分布统计】")
        print(f"{'指标':<20} {'ASMC':<15} {'Gaussian VA':<15}")
        print("-" * 50)
        print(f"{'厚度平均标准差(nm)':<20} {eval_smc['avg_std']:.2f} {eval_pinn['avg_std']:.2f}")
        print(f"{'孔隙率均值':<20} {eval_smc['f_mean']:.4f} {eval_pinn['f_mean']:.4f}")
        print(f"{'孔隙率标准差':<20} {eval_smc['f_std']:.4f} {eval_pinn['f_std']:.4f}")
        # R1-2：经验 2-sigma 覆盖率（标定正确时应接近 0.95）
        print(f"{'厚度2σ覆盖率':<20} {eval_smc['thickness_coverage_2sigma']:.2f} "
              f"{eval_pinn['thickness_coverage_2sigma']:.2f}")
        print(f"{'孔隙率2σ覆盖率':<20} {eval_smc['porosity_coverage_2sigma']:.2f} "
              f"{eval_pinn['porosity_coverage_2sigma']:.2f}")

        print("\n【ASMC特有指标】")
        final_ess = res_smc['ess_history'][-1] if res_smc['ess_history'] else 0
        print(f"最终有效样本数(ESS): {final_ess:.0f}/{Config().N_PARTICLES_SMC}")

        print("\n【Gaussian VA特有指标】")
        final_loss = res_pinn['loss_history'][-1] if res_pinn['loss_history'] else 0
        print(f"最终训练损失: {final_loss:.6f}")
        print("\n" + "=" * 80 + "\n")

        return {
            'sigma': sigma,
            'spectral_rmse_smc': spectral_rmse_smc,
            'spectral_rmse_pinn': spectral_rmse_pinn,
            'thickness_rel_err_smc': eval_smc['thickness_rel_err'],
            'thickness_rel_err_pinn': eval_pinn['thickness_rel_err'],
            'porosity_rel_err_smc': eval_smc['porosity_rel_err'],
            'porosity_rel_err_pinn': eval_pinn['porosity_rel_err'],
            'runtime_smc': runtime_smc,
            'runtime_pinn': runtime_pinn
        }


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

    @staticmethod
    def _nice_step(vmin, vmax, target):
        """按 1 / 2 / 5 × 10^k 规则取整刻度间隔，保证刻度位置整齐、标签位数最短。"""
        raw = (vmax - vmin) / max(int(target), 1)
        if raw <= 0:
            return 1.0
        exp = np.floor(np.log10(raw))
        cands = [m * (10 ** exp) for m in (1.0, 2.0, 5.0)] + [10.0 * (10 ** exp)]
        return min(cands, key=lambda s: abs(s - raw))

    def _set_ticks(self, ax, axis='y', target=6, interval=None):
        """设置等间距的数值刻度（审稿人要求：图 1 需给出坐标轴数值标签）。

        axis    : 'x' 横轴 / 'y' 纵轴
        target  : 期望刻度档数，用于自动取整间隔（未给 interval 时生效）
        interval: 固定间隔；波长轴用 20 nm，保证标签为整数

        标签格式由间隔量级决定：间隔 >= 1 用整数 '%d'，否则用 %.{n}f，
        n = -floor(log10(step))，因此 0.02 -> '0.02'、2e-5 -> '0.00002'。
        """
        lim = ax.get_xlim() if axis == 'x' else ax.get_ylim()
        step = float(interval) if interval is not None else self._nice_step(lim[0], lim[1], target)
        start = np.ceil(lim[0] / step) * step
        ticks = np.arange(start, lim[1] + 0.5 * step, step)
        # 消除浮点产生的 -0.0，避免标签显示成 "-0.00"
        ticks = np.where(np.abs(ticks) < 1e-15, 0.0, ticks)
        ndec = max(0, int(-np.floor(np.log10(step))))
        fmt = '%d' if ndec == 0 else f'%.{ndec}f'
        labels = [fmt % t for t in ticks]
        if axis == 'x':
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

    def plot_spectral_ab_test(self, data, res_smc, res_pinn, physics, sigma, seed: int = 2024):
        """
        图1: 光谱拟合深度分析 (Double-Decker Layout with Zoom-Lens)
        """
        wl = data['wavelengths'].cpu().numpy()
        R_obs = data['R_obs'][0].cpu().numpy()

        def get_pred(res):
            # 用独立 RandomState：原实现未设种子，每次重跑 μ/σ 阴影都会漂移。
            idx = np.random.RandomState(seed).choice(len(res['particles']), 300)
            ps = torch.tensor(res['particles'][idx], device=physics.device, dtype=torch.float32)
            with torch.no_grad():
                R = physics.tmm_reflectance(ps[:, :5], ps[:, 5:10], ps[:, 10:], data['wavelengths'], 0.0)
            return R.mean(0).cpu().numpy(), R.std(0).cpu().numpy()

        mu_smc, std_smc = get_pred(res_smc)
        mu_pinn, std_pinn = get_pred(res_pinn)

        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1], hspace=0.1, wspace=0.15)

        # === ASMC panel ===
        ax_main_l = fig.add_subplot(gs[0, 0])
        ax_res_l = fig.add_subplot(gs[1, 0], sharex=ax_main_l)

        ax_main_l.plot(wl, mu_smc, color=self.colors['smc'], lw=2, label='Posterior Mean')
        ax_main_l.fill_between(wl, mu_smc - 2 * std_smc, mu_smc + 2 * std_smc, color=self.colors['smc_l'], alpha=0.6,
                               label=r'Posterior predictive $\pm 2\sigma$')
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
        axins_l.set_xticks([])
        axins_l.set_yticks([])
        for spine in axins_l.spines.values():
            spine.set_color(self.colors['smc'])
            spine.set_linewidth(1.5)
        mark_inset(ax_main_l, axins_l, loc1=2, loc2=4, fc="none", ec="0.5", ls=':')

        # 原实现写作 res_smc = R_obs - mu_smc，会覆盖同名函数参数（含 'particles' 的 dict），
        # 此后若再访问 res_smc['particles'] 即崩溃；改名以消除该隐患。
        resid_smc = R_obs - mu_smc
        ax_res_l.plot(wl, resid_smc, color=self.colors['smc'], lw=1)
        ax_res_l.fill_between(wl, -2 * std_smc, 2 * std_smc, color='gray', alpha=0.1)
        ax_res_l.axhline(0, color='black', lw=0.8, ls='--')

        self._style_axis(ax_main_l)
        self._style_axis(ax_res_l)
        ax_main_l.set_ylabel('Reflectance ($R$)', fontsize=11)
        ax_res_l.set_ylabel(r'Residual ($\Delta R$)', fontsize=11)
        ax_res_l.set_xlabel('Wavelength (nm)', fontsize=11)
        ax_main_l.set_title("ASMC Inversion", color=self.colors['smc'], fontsize=12, pad=10)
        self._add_panel_label(ax_main_l, '(a)')

        # === Gaussian VA panel ===
        ax_main_r = fig.add_subplot(gs[0, 1], sharey=ax_main_l)
        ax_res_r = fig.add_subplot(gs[1, 1], sharex=ax_main_r, sharey=ax_res_l)

        ax_main_r.plot(wl, mu_pinn, color=self.colors['pinn'], lw=2, label='Posterior Mean')
        ax_main_r.fill_between(wl, mu_pinn - 2 * std_pinn, mu_pinn + 2 * std_pinn, color=self.colors['pinn_l'],
                               alpha=0.6)
        ax_main_r.scatter(wl[::4], R_obs[::4], color=self.colors['obs'], s=15, alpha=0.5, marker='o')

        # Gaussian VA zoom inset
        axins_r = inset_axes(ax_main_r, width="40%", height="35%", loc='lower left',
                             bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax_main_r.transAxes)
        axins_r.plot(wl[zoom_mask], mu_pinn[zoom_mask], color=self.colors['pinn'], lw=2)
        axins_r.fill_between(wl[zoom_mask], (mu_pinn - 2 * std_pinn)[zoom_mask], (mu_pinn + 2 * std_pinn)[zoom_mask],
                             color=self.colors['pinn_l'], alpha=0.6)
        axins_r.scatter(wl[zoom_mask], R_obs[zoom_mask], color=self.colors['obs'], s=20, alpha=0.6)
        axins_r.set_xticks([])
        axins_r.set_yticks([])
        for spine in axins_r.spines.values():
            spine.set_color(self.colors['pinn'])
            spine.set_linewidth(1.5)
        mark_inset(ax_main_r, axins_r, loc1=2, loc2=4, fc="none", ec="0.5", ls=':')

        # 同上：改名，避免覆盖函数参数 res_pinn
        resid_pinn = R_obs - mu_pinn
        ax_res_r.plot(wl, resid_pinn, color=self.colors['pinn'], lw=1)
        ax_res_r.fill_between(wl, -2 * std_pinn, 2 * std_pinn, color='gray', alpha=0.1)
        ax_res_r.axhline(0, color='black', lw=0.8, ls='--')

        self._style_axis(ax_main_r)
        self._style_axis(ax_res_r)
        ax_res_r.set_xlabel('Wavelength (nm)', fontsize=11)
        ax_main_r.set_title("Gaussian VA Inversion", color=self.colors['pinn'], fontsize=12, pad=10)
        self._add_panel_label(ax_main_r, '(b)')

        # ---- 数值刻度 ----
        # 横轴：波长固定 20 nm 间隔（标签为整数 nm）。刻度设在下方残差面板上，
        #       因与主图共享 x 轴，主图会同步得到刻度；再用 setp(..., visible=False)
        #       只隐藏主图的标签，保持双层图整洁（直接 set_xticklabels([]) 会
        #       连带清掉共享轴、导致两个面板都没有刻度）。
        # 纵轴：主图 6 档、残差图 5 档，间隔按 1/2/5×10^k 规则自动取整；
        #       同理，纵轴刻度设在左列，右列因 sharey 自动同步。
        for ax_res, ax_main in ((ax_res_l, ax_main_l), (ax_res_r, ax_main_r)):
            self._set_ticks(ax_res, axis='x', interval=20.0)
            plt.setp(ax_main.get_xticklabels(), visible=False)
        self._set_ticks(ax_main_l, axis='y', target=6)
        self._set_ticks(ax_res_l, axis='y', target=5)

        handles, labels = ax_main_l.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=10)

        plt.suptitle(rf'Spectral Reconstruction Analysis (Noise Level $\sigma={sigma}$)', y=1.02, fontsize=14,
                     fontweight='bold')
        plt.savefig(OUTPUT_DIR / f'Fig1_Spectral_Analysis_Sigma_{sigma}_clean.png', bbox_inches='tight', dpi=300)
        plt.show()

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

        plt.suptitle(rf'Parameter Coupling & Joint Distribution (Noise $\sigma={sigma}$)', y=0.98, fontsize=14,
                     fontweight='bold')
        plt.savefig(OUTPUT_DIR / f'Fig2_Joint_Topology_Sigma_{sigma}_clean.png', bbox_inches='tight', dpi=300)
        plt.show()

    def add_metrics(self, metrics):
        self.all_metrics.append(metrics)

    def plot_slopegraph_summary(self):
        """
        Supplementary performance summary (Slope-Dumbbell Chart)
        """
        noise_labels = ['Low Noise\n' + r'($\sigma=10^{-4}$)',
                        'High Noise\n' + r'($\sigma=5\times10^{-3}$)']
        x_pos = [0, 1]
        metrics_sorted = sorted(self.all_metrics, key=lambda x: x['sigma'])

        plot_data = {
            'rmse': {
                'title': 'Spectral Accuracy (RMSE)', 'unit': 'Log Scale', 'is_log': True,
                'smc': [m['spectral_rmse_smc'] for m in metrics_sorted],
                'pinn': [m['spectral_rmse_pinn'] for m in metrics_sorted]
            },
            'thickness': {
                'title': 'Thickness Error', 'unit': 'Relative Error (%)', 'is_log': False,
                'smc': [m['thickness_rel_err_smc'] for m in metrics_sorted],
                'pinn': [m['thickness_rel_err_pinn'] for m in metrics_sorted]
            },
            'porosity': {
                'title': 'Porosity Error', 'unit': 'Relative Error (%)', 'is_log': False,
                'smc': [m['porosity_rel_err_smc'] for m in metrics_sorted],
                'pinn': [m['porosity_rel_err_pinn'] for m in metrics_sorted]
            },
            'time': {
                'title': 'Computing Time', 'unit': 'Seconds', 'is_log': False,
                'smc': [m['runtime_smc'] for m in metrics_sorted],
                'pinn': [m['runtime_pinn'] for m in metrics_sorted]
            }
        }

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

        for idx, (key, item) in enumerate(plot_data.items()):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            y_smc = item['smc']
            y_pinn = item['pinn']

            ax.plot(x_pos, y_smc, '-', color=self.colors['smc'], lw=2, alpha=0.8, label='ASMC' if idx == 0 else "")
            ax.plot(x_pos, y_pinn, '-', color=self.colors['pinn'], lw=2, alpha=0.8, label='Gaussian VA' if idx == 0 else "")

            ax.scatter(x_pos, y_smc, c=self.colors['smc'], s=100, zorder=5, edgecolors='white', lw=1.5)
            ax.scatter(x_pos, y_pinn, c=self.colors['pinn'], s=100, zorder=5, marker='D', edgecolors='white', lw=1.5)

            for i, (vs, vp) in enumerate(zip(y_smc, y_pinn)):
                offset_scale = 0.6 if item['is_log'] else 1.0
                s_top = vs >= vp
                offset_s = (12 if s_top else -20) * offset_scale
                offset_p = (-20 if s_top else 12) * offset_scale
                fmt = '{:.1e}' if item['is_log'] else '{:.1f}'
                ax.annotate(fmt.format(vs), (i, vs), xytext=(0, offset_s), textcoords='offset points',
                            ha='center', color=self.colors['smc'], fontweight='bold', fontsize=9)
                ax.annotate(fmt.format(vp), (i, vp), xytext=(0, offset_p), textcoords='offset points',
                            ha='center', color=self.colors['pinn'], fontweight='bold', fontsize=9)

            self._style_axis(ax, grid=True)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(noise_labels, fontweight='bold')
            ax.set_xlim(-0.2, 1.2)

            if item['is_log']:
                ax.set_yscale('log')
                ymin = min(min(y_smc), min(y_pinn)) * 0.4
                ymax = max(max(y_smc), max(y_pinn)) * 6.0
                ax.set_ylim(ymin, ymax)
            else:
                ymin = min(min(y_smc), min(y_pinn)) * 0.8
                ymax = max(max(y_smc), max(y_pinn)) * 1.2
                ax.set_ylim(ymin, ymax)

            ax.set_title(item['title'], loc='left', fontsize=12, fontweight='bold')
            ax.set_ylabel(item['unit'], fontsize=10, color=self.colors['text'])
            self._add_panel_label(ax, f"({chr(97 + idx + 4)})")

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=self.colors['smc'], lw=2, marker='o'),
                        Line2D([0], [0], color=self.colors['pinn'], lw=2, marker='D')]
        fig.legend(custom_lines, ['ASMC', 'Gaussian VA'],
                   loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=11)

        fig.suptitle('Noise-level comparison', fontsize=16, fontweight='bold', y=0.99)
        plt.savefig(OUTPUT_DIR / 'Supplementary_Performance_Trends.png', bbox_inches='tight', dpi=300)
        plt.show()


# ============================================================
# 后验相关性工具
# ============================================================

def _safe_corrcoef(particles):
    """相关矩阵；对退化为常数的维度（std=0）相关记 0，而不是 NaN。

    VA 的对角后验训练后部分维度（dn、f）std 会塌缩到 0，np.corrcoef 对这些维度
    返回 NaN 并触发 RuntimeWarning —— 这正是运行结果里出现 "corr(d1,dn2)=+nan" 的原因。
    """
    X = np.asarray(particles, dtype=np.float64)
    good = X.std(axis=0, ddof=0) > 0
    corr = np.zeros((X.shape[1], X.shape[1]))
    idx = np.where(good)[0]
    if len(idx) >= 2:
        corr[np.ix_(idx, idx)] = np.corrcoef(X[:, idx].T)
    np.fill_diagonal(corr, 1.0)
    return corr, int((~good).sum())


# ============================================================
# 主执行流程
# ============================================================

def run_noise_level_comparison(noise_levels: List[float] = [1e-4, 5e-3], seeds: List[int] = None):
    """R1-5：每个噪声水平可在多个独立种子下重复，并报告均值±标准差。

    seeds 默认为 [2024]，即与原实现一致的单种子行为；传入多个种子时，
    图表使用最后一个种子的结果渲染（每个种子的图各不相同，全部渲染会
    产生过多文件），而数值统计覆盖全部种子，在末尾输出"均值 ± 标准差"汇总。
    """
    seeds = list(seeds) if seeds else [2024]
    print("=" * 90)
    print(f"Generating manuscript figures and detailed console output (seeds={seeds})")
    print("=" * 90)

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
                                                       runtime_smc, runtime_pinn, seed)
            plotter.add_metrics(metrics)
            trials.append({'sigma': sigma, 'seed': seed,
                           'smc': Evaluator.evaluate(res_smc['particles'], data['true_params'], 'ASMC'),
                           'va': Evaluator.evaluate(res_pinn['particles'], data['true_params'], 'Gaussian VA')})

            last_data, last_smc, last_pinn, last_seed = data, res_smc, res_pinn, seed

        print("   Rendering high-resolution figures...")
        plotter.plot_spectral_ab_test(last_data, last_smc, last_pinn, physics, sigma, last_seed)
        plotter.plot_joint_distribution(last_smc, last_pinn, last_data['true_params'], sigma)
        # R1-4：后验参数相关性（对角 Gaussian VA 无法表达的部分）
        for tag, particles in [('ASMC', last_smc['particles']),
                               ('Gaussian VA', last_pinn['particles'])]:
            # 用 _safe_corrcoef 替代 np.corrcoef：VA 部分维度 std 塌缩为 0 时，
            # np.corrcoef 会返回 NaN（运行结果中的 "corr(d1,dn2)=+nan"）
            corr, n_degenerate = _safe_corrcoef(particles)
            pairs = sorted(((abs(corr[i, j]), corr[i, j], i, j)
                            for i in range(11) for j in range(i + 1, 11)), reverse=True)
            print(f"\n【R1-4 后验相关性 · {tag}】最强关联参数对:")
            if n_degenerate:
                print(f"    注：{n_degenerate} 个维度后验 std 已塌缩为 0（常数），其相关按 0 计")
            for _, c, i, j in pairs[:5]:
                print(f"    corr({LAYER_NAMES[i]:>10s}, {LAYER_NAMES[j]:>10s}) = {c:+.3f}")
            print(f"    d1-f 相关性 = {corr[0, 10]:+.3f}")

    # R1-5：多种子重复下的均值 ± 标准差
    if len(seeds) > 1:
        print("\n" + "=" * 88)
        print(f"R1-5 多种子统计（每噪声水平 {len(seeds)} 个独立噪声实现）")
        print("=" * 88)
        for sigma in sorted({t['sigma'] for t in trials}):
            group = [t for t in trials if t['sigma'] == sigma]
            print(f"\n--- sigma = {sigma:g} ---")
            print(f"{'指标':<24}{'ASMC (mean±std)':<26}{'Gaussian VA (mean±std)':<26}")
            print("-" * 76)
            for label, key in [('厚度相对误差(%)', 'thickness_rel_err'),
                               ('孔隙率相对误差(%)', 'porosity_rel_err'),
                               ('厚度平均std(nm)', 'avg_std'),
                               ('厚度2σ覆盖率', 'thickness_coverage_2sigma')]:
                vs = np.array([t['smc'][key] for t in group], dtype=np.float64)
                vv = np.array([t['va'][key] for t in group], dtype=np.float64)
                print(f"{label:<24}{vs.mean():.3f} ± {vs.std(ddof=1):<14.3f}"
                      f"{vv.mean():.3f} ± {vv.std(ddof=1):.3f}")

    print("   Rendering summary slopegraph...")
    plotter.plot_slopegraph_summary()
    print("\nAll figures generated successfully.")


def _print_index_guard_report(cfg: Config, physics: PhysicsModel):
    """R1-1：折射率钳位的定量依据——打印波段内的实际取值范围。

    审稿人要求给出这些钳位的物理依据。这里直接给出启用/关闭钳位时各材料在
    550-720 nm 内的折射率跨度：若跨度为 0，说明色散被钳位完全替换成常数，
    钳位就不再是"物理约束"而是"固定取值"，必须在论文中给出依据或移除。
    多孔层表面折射率额外给出孔隙率先验两端 f=0.30 / 0.50 处的取值与跨度，
    用于判断该钳位是否在整个先验区间内饱和。
    """
    dev = torch.device(cfg.DEVICE)
    wl = torch.linspace(*cfg.WAVELENGTH_RANGE, cfg.N_WAVELENGTHS, device=dev)
    n_hf = physics.sellmeier_hfo2(wl)
    n_ta = physics.sellmeier_ta2o5(wl)
    n_uld = physics.sellmeier_sio2(wl)
    n_surf_lo = physics.moth_eye_effective_n(n_uld, torch.full_like(wl, 0.30))
    n_surf_hi = physics.moth_eye_effective_n(n_uld, torch.full_like(wl, 0.50))

    def _span(x):
        return float(x.max() - x.min())

    print(f"    钳位报告 (ENABLE_INDEX_CLAMPS={cfg.ENABLE_INDEX_CLAMPS}):")
    print(f"      HfO2   n ∈ [{float(n_hf.min()):.5f}, {float(n_hf.max()):.5f}]  跨度 = {_span(n_hf):.5f}")
    print(f"      Ta2O5  n ∈ [{float(n_ta.min()):.5f}, {float(n_ta.max()):.5f}]  跨度 = {_span(n_ta):.5f}")
    print(f"      多孔层表面 n(f=0.30) = {float(n_surf_lo.mean()):.5f} | "
          f"n(f=0.50) = {float(n_surf_hi.mean()):.5f} | "
          f"先验两端差 = {float(n_surf_hi.mean() - n_surf_lo.mean()):.5f}")


# R1-1 敏感性扫描的工况表：(ENABLE_INDEX_CLAMPS, DESIGN_WAVELENGTH_WEIGHT)
#   (True, 10.0) 基线——与主流程 Table 3 同一设定，作为比对的参照列；
#   (True, 1.0)  仅改加权：单因素隔离 632.8nm 十倍加权的影响；
#   (False, 10.0) 仅改钳位：单因素隔离折射率钳位的影响（同时支撑 R1-6 孔隙率敏感性）。
# 删去 (False, 1.0) 这个"加权×钳位"交互角：它属二阶效应，且会与两个单因素列重复。
R1_1_VARIANTS: Tuple[Tuple[bool, float], ...] = ((True, 10.0), (True, 1.0), (False, 10.0))


def run_r1_1_model_choice_sensitivity(noise_levels: List[float] = [1e-4, 5e-3],
                                      seeds: List[int] = None,
                                      variants: Tuple[Tuple[bool, float], ...] = R1_1_VARIANTS):
    """R1-1：量化"设计波长加权"与"折射率钳位"两个特定选择对 Table 3 数值的影响。

    审稿人 1 要求：说明 632.8 nm 十倍加权的依据、澄清两法是否用同一加权目标、
    并检验 Table 3 的具体数值对这些特定选择的敏感性、说明折射率钳位的依据。
    本函数按 R1_1_VARIANTS 中列出的工况（钳位开关, 权重倍数）× 噪声水平 逐个重跑，
    两法在同一组设置下各自重跑，输出可直接抄进回复信的对照表与极差。

    基线列被保留在工况表内，是为了让"极差"可分解为两个单因素效应
    （加权：基线 vs 1×；钳位：基线 vs 钳位关），避免把两个因素混在一起。

    扫描显式把 VA_SHARED_LIKELIHOOD 置 True（与 Config 默认一致），确保两法
    使用完全相同的加权高斯似然，这正是 R1-1"两法是否使用同一加权目标"所要求的
    对照条件。若把 Config 改回 False，可用本函数对照"两法目标不同"时的旧结果。
    """
    seeds = list(seeds) if seeds else [2024]
    print("=" * 96)
    print("R1-1 模型选择敏感性：设计波长加权 × 折射率钳位 × 噪声水平  （两法共用同一加权似然）")
    print("=" * 96)

    metric_spec = [('厚度相对误差(%)', 'thickness_rel_err'),
                   ('孔隙率相对误差(%)', 'porosity_rel_err'),
                   ('厚度平均std(nm)', 'avg_std'),
                   ('厚度RMSE(nm)', 'thickness_rmse'),
                   ('厚度2σ覆盖率', 'thickness_coverage_2sigma')]
    rows = []

    for use_clamps, wfac in variants:
        cfg = Config()
        cfg.ENABLE_INDEX_CLAMPS = use_clamps
        cfg.USE_WEIGHTED_LIKELIHOOD = (wfac != 1.0)
        cfg.DESIGN_WAVELENGTH_WEIGHT = wfac
        cfg.VA_SHARED_LIKELIHOOD = True
        physics = PhysicsModel(cfg)

        tag = f"钳位={'开' if use_clamps else '关'} | 632.8nm权重={wfac:g}×"
        print(f"\n--- {tag} ---")
        _print_index_guard_report(cfg, physics)

        for sigma in noise_levels:
            for seed in seeds:
                cfg.SIGMA_OBS = sigma
                torch.manual_seed(seed)
                np.random.seed(seed)
                data = DataGenerator(cfg, physics).generate(seed)

                res_smc = SMCMethod(cfg, physics).run(data, seed)
                va = GaussianVA(cfg).to(torch.device(cfg.DEVICE))
                res_va = va.run(data, physics, seed)

                ev_smc = Evaluator.evaluate(res_smc['particles'], data['true_params'], 'ASMC')
                ev_va = Evaluator.evaluate(res_va['particles'], data['true_params'], 'Gaussian VA')
                for ev in (ev_smc, ev_va):
                    print(f"    σ={sigma:g} seed={seed} · {ev['method']:<12} "
                          f"厚度误差 {ev['thickness_rel_err']:7.4f}% | "
                          f"孔隙率误差 {ev['porosity_rel_err']:7.4f}% | "
                          f"avg_std {ev['avg_std']:7.4f} | "
                          f"2σ覆盖 {ev['thickness_coverage_2sigma']:.2f}")
                    rows.append({'clamps': use_clamps, 'weight': wfac, 'sigma': sigma,
                                 'seed': seed, 'method': ev['method'],
                                 **{key: ev[key] for _, key in metric_spec}})

    print("\n" + "=" * 96)
    print(f"R1-1 汇总：同一指标在 {len(variants)} 种模型选择下的取值与极差"
          f"（Table 3 数值对这些选择的敏感性）")
    print("=" * 96)
    head = [f"钳位{'开' if c else '关'}/{w:g}×" for c, w in variants]
    print(f"{'σ':<8}{'方法':<14}{'指标':<20}" + "".join(f"{h:<16}" for h in head) + "极差")
    print("-" * 110)
    for sigma in noise_levels:
        for method in ('ASMC', 'Gaussian VA'):
            for label, key in metric_spec:
                vals = [float(np.mean([r[key] for r in rows
                                       if r['sigma'] == sigma and r['method'] == method
                                       and r['clamps'] == c and r['weight'] == w]))
                        for c, w in variants]
                spread = max(vals) - min(vals)
                print(f"{sigma:<8g}{method:<14}{label:<20}"
                      + "".join(f"{v:<16.4f}" for v in vals) + f"{spread:.4f}")
    print("\nR1-1 敏感性扫描完成。")


if __name__ == "__main__":
    # 固定输入：两个噪声水平 × 单种子 2024，与原实现一致。
    # R1-5 多种子重复：把 seeds 改为 [2024, 2025, 2026] 后重跑即可，
    # 主流程会自动追加"均值 ± 标准差"汇总。
    run_noise_level_comparison(noise_levels=[1e-4, 5e-3], seeds=[2024])
    # R1-1：Table 3 数值对"设计波长加权 / 折射率钳位"这两个特定选择的敏感性扫描。
    run_r1_1_model_choice_sensitivity(noise_levels=[1e-4, 5e-3], seeds=[2024])
