"""
Aşama 3c: QUBO Formülasyonu — Phase-Only One-Hot + Amplitude Tapered
Yöntem: PyQUBO sembolik değişkenler + lookup table

Faz: One-hot kodlama → cos/sin katsayıları sabit sayı → QUBO kuadratik kalır.
Genlik: Daha büyük elemanlar için Taylor penceresi (sabit, optimize edilmez).
         Genlik optimizasyonu ileride eklenir (Senaryo 5+).

Değişken sayısı: N × n_phase (N=8 için 64, N=16 için 128)
Kurulum süresi: ~0.2s (N=8), ~2s (N=16)
SA çözüm süresi: ~0.2s (N=8), ~1s (N=16)
"""

import numpy as np
import time
from pyqubo import Array
from dwave.samplers import SimulatedAnnealingSampler
from src.array_factor import ArrayConfig, beam_metrics


def build_qubo_phase_onehot(config: ArrayConfig,
                             theta_sidelobe_samples: np.ndarray = None,
                             lambda_sll: float = 0.2,
                             lambda_null: float = 5.0,
                             penalty: float = 20.0,
                             amplitudes: np.ndarray = None):
    """
    Phase-only one-hot QUBO kurar.
    
    Parametreler:
        amplitudes: Sabit genlik vektörü (None ise tüm elemanlar 1.0)
        penalty   : One-hot kısıt ceza ağırlığı
    
    Dönüş:
        qubo (dict): SA'ya verilecek {(v1,v2): katsayı} sözlüğü
        model     : PyQUBO model (decode için)
        config    : Aynı config (decode için)
    """
    N = config.N
    n_phase = config.n_phase_levels
    phase_levels = config.phase_levels_deg
    kd = 2 * np.pi * config.d_over_lambda

    if amplitudes is None:
        amplitudes = np.ones(N)

    if theta_sidelobe_samples is None:
        all_t = np.arange(-90, 91, 10)
        theta_sidelobe_samples = all_t[np.abs(all_t - config.theta_target) > 15]
    n_sl = max(len(theta_sidelobe_samples), 1)
    n_nl = max(len(config.theta_nulls), 1)

    # One-hot faz değişkenleri: x[n][k] = eleman n, faz seviyesi k
    x = Array.create('x', shape=(N, n_phase), vartype='BINARY')

    def af_power_symbolic(theta_deg):
        """Belirli bir açıda |AF|² sembolik ifadesi."""
        theta_rad = np.deg2rad(theta_deg)
        re_total, im_total = 0, 0
        for n in range(N):
            phase_n = n * kd * np.sin(theta_rad)
            cos_n = np.cos(phase_n)
            sin_n = np.sin(phase_n)
            # One-hot: cos(ψ_n) = Σ_k cos(phase_levels[k]) * x[n,k]
            cos_w = sum(float(np.cos(np.deg2rad(phase_levels[k]))) * x[n, k]
                        for k in range(n_phase))
            sin_w = sum(float(np.sin(np.deg2rad(phase_levels[k]))) * x[n, k]
                        for k in range(n_phase))
            A_n = float(amplitudes[n])
            # w_n * steering_n: reel ve imajiner
            re_total += A_n * (cos_w * cos_n - sin_w * sin_n)
            im_total += A_n * (cos_w * sin_n + sin_w * cos_n)
        return re_total ** 2 + im_total ** 2

    # ── Hamiltonyen ──
    # Ana lob: maksimize → negatif ekle
    H = -1.0 * af_power_symbolic(config.theta_target)

    # Yan lob bölgesi
    for theta in theta_sidelobe_samples:
        H = H + (lambda_sll / n_sl) * af_power_symbolic(theta)

    # Null bölgesi
    for theta in config.theta_nulls:
        H = H + (lambda_null / n_nl) * af_power_symbolic(theta)

    # One-hot kısıtı: her eleman için tam olarak bir faz seçili olmalı
    for n in range(N):
        H = H + penalty * (sum(x[n, k] for k in range(n_phase)) - 1) ** 2

    # Compile → QUBO
    model = H.compile()
    qubo, offset = model.to_qubo()

    return qubo, model, config


def decode_phase_onehot(sample: dict, config: ArrayConfig,
                         amplitudes: np.ndarray = None):
    """One-hot çözümünü fiziksel amplitude/phase dizisine çevir."""
    N = config.N
    n_phase = config.n_phase_levels
    phase_levels = config.phase_levels_deg

    if amplitudes is None:
        amplitudes = np.ones(N)

    phases_deg = np.zeros(N)
    n_violations = 0

    for n in range(N):
        active = [k for k in range(n_phase)
                  if sample.get(f'x[{n}][{k}]', 0) == 1]
        if len(active) == 1:
            phases_deg[n] = phase_levels[active[0]]
        elif len(active) > 1:
            phases_deg[n] = phase_levels[active[0]]
            n_violations += 1
        else:
            phases_deg[n] = 0.0
            n_violations += 1

    return amplitudes.copy(), phases_deg, n_violations


def solve_sa_bw(config: ArrayConfig,
                thetas_deg: np.ndarray,
                lambda_sll: float = 0.2,
                lambda_null: float = 5.0,
                main_lobe_weight: float = 1.0,
                amplitudes: np.ndarray = None,
                num_reads: int = 500,
                num_sweeps: int = 2000,
                verbose: bool = True):
    """
    Phase-only one-hot QUBO + SA çözücü.
    main_lobe_weight: ana lob ceza ağırlığı (penalty * ile değil, H'ye gömülü)
    """
    from src.solvers import SolverResult

    if verbose:
        print(f"\n  [SA] QUBO kuruluyor... N={config.N}, "
              f"{config.n_phase_levels} faz seviyesi, "
              f"λ_sll={lambda_sll}, λ_null={lambda_null}")

    t0 = time.time()
    qubo, model, _ = build_qubo_phase_onehot(
        config,
        lambda_sll=lambda_sll,
        lambda_null=lambda_null,
        amplitudes=amplitudes,
    )
    t_build = time.time() - t0

    n_vars = len(set(v for pair in qubo.keys() for v in pair))
    if verbose:
        print(f"  [SA] {n_vars} değişken | {len(qubo)} terim | {t_build:.2f}s kurulum")

    sampler = SimulatedAnnealingSampler()
    t1 = time.time()
    sampleset = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps)
    t_sa = time.time() - t1

    best = sampleset.first
    if amplitudes is None:
        amplitudes = np.ones(config.N)
    amps, phases, n_viol = decode_phase_onehot(best.sample, config, amplitudes)
    m = beam_metrics(config, amps, phases, thetas_deg)

    result = SolverResult(
        method="SA",
        amplitudes=amps,
        phases_deg=phases,
        sll_db=m["sll_db"],
        hpbw_deg=m["hpbw_deg"],
        main_lobe_deg=m["main_lobe_deg"],
        null_depths=m["null_depths"],
        best_energy=best.energy,
        solve_time_sec=t_build + t_sa,
        n_violations=n_viol,
        lambda_sll=lambda_sll,
        lambda_null=lambda_null,
    )

    if verbose:
        print(result)

    return result
