"""
Aşama 3b: Lambda Ağırlık Tarama (Grid Search)
QUBO maliyet fonksiyonundaki λ1 (SLL) ve λ2 (null) ağırlıklarını
sistemli biçimde tarayarak en iyi performans/kararlılık dengesini bulur.

Neden gerekli?
─────────────────
λ çok küçükse  → optimizer sadece ana lobu önemser, yan loblar bastırılmaz.
λ çok büyükse  → optimizer yan lob/null'a o kadar odaklanır ki ana lobu kaybeder.
Doğru aralık, normalize edilmiş enerji terimlerinin BİRBİRİNE YAKIN
büyüklükte olduğu bölgedir (tipik olarak 0.05 - 0.5 arası, sisteme göre değişir).

Her λ kombinasyonu için BİRDEN FAZLA SA çalıştırması yapılır (tekrar sayısı),
çünkü Simulated Annealing stokastiktir — tek çalıştırma yanıltıcı olabilir.
"""

import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler
from src.array_factor import ArrayConfig, beam_metrics
from src.qubo_formulation import build_qubo_onehot, decode_solution


def evaluate_lambda(config: ArrayConfig,
                    lambda_sll: float,
                    lambda_null: float,
                    thetas_deg: np.ndarray,
                    n_repeats: int = 5,
                    num_reads: int = 200,
                    num_sweeps: int = 1000) -> dict:
    """
    Belirli bir (lambda_sll, lambda_null) çifti için n_repeats kez SA çalıştırır,
    ortalama ve std sapma istatistiklerini döndürür.
    
    Dönüş (dict):
        main_lobe_error_mean/std : Hedeften sapma (derece)
        sll_db_mean/std          : SLL ortalaması
        violations_mean          : Ortalama kısıt ihlali sayısı
        best_energy              : En düşük QUBO enerjisi (n_repeats içinde)
    """
    H, variables = build_qubo_onehot(config, lambda_sll=lambda_sll, lambda_null=lambda_null)
    qubo, offset = H.compile().to_qubo()
    sampler = SimulatedAnnealingSampler()

    errors, slls, violations_list, energies = [], [], [], []

    for _ in range(n_repeats):
        sampleset = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps)
        best = sampleset.first
        amps, phases, viol = decode_solution(best.sample, variables)
        m = beam_metrics(config, amps, phases, thetas_deg)

        errors.append(abs(m["main_lobe_deg"] - config.theta_target))
        slls.append(m["sll_db"])
        violations_list.append(viol)
        energies.append(best.energy)

    return {
        "lambda_sll": lambda_sll,
        "lambda_null": lambda_null,
        "main_lobe_error_mean": np.mean(errors),
        "main_lobe_error_std": np.std(errors),
        "sll_db_mean": np.mean(slls),
        "sll_db_std": np.std(slls),
        "violations_mean": np.mean(violations_list),
        "best_energy": np.min(energies),
    }


def grid_search(config: ArrayConfig,
                thetas_deg: np.ndarray,
                lambda_sll_range: list = None,
                lambda_null_range: list = None,
                n_repeats: int = 5,
                verbose: bool = True) -> pd.DataFrame:
    """
    Verilen λ1, λ2 aralıklarını tarar, her kombinasyon için evaluate_lambda
    çağırır, sonuçları DataFrame olarak döndürür.
    
    Eğer config.theta_nulls boşsa, lambda_null taranmaz (sabit 0 kalır).
    """
    if lambda_sll_range is None:
        lambda_sll_range = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    if config.theta_nulls:
        if lambda_null_range is None:
            lambda_null_range = [0.0, 1.0, 2.0, 5.0, 10.0]
    else:
        lambda_null_range = [0.0]  # Null hedefi yoksa taramaya gerek yok

    results = []
    total = len(lambda_sll_range) * len(lambda_null_range)
    count = 0

    for l_sll in lambda_sll_range:
        for l_null in lambda_null_range:
            count += 1
            if verbose:
                print(f"  [{count}/{total}] λ_sll={l_sll:.3f}, λ_null={l_null:.2f} ... ", end="", flush=True)

            res = evaluate_lambda(config, l_sll, l_null, thetas_deg, n_repeats=n_repeats)
            results.append(res)

            if verbose:
                print(f"hata={res['main_lobe_error_mean']:.1f}±{res['main_lobe_error_std']:.1f}° | "
                      f"SLL={res['sll_db_mean']:.1f} dB | ihlal={res['violations_mean']:.1f}")

    df = pd.DataFrame(results)
    return df


def find_best_lambda(df: pd.DataFrame,
                     error_weight: float = 2.0,
                     sll_weight: float = 1.0) -> dict:
    """
    Grid search sonuçlarından en iyi (lambda_sll, lambda_null) çiftini seçer.
    
    Skor = error_weight · (ortalama_hata / max_hata) + sll_weight · (SLL_normalize)
    Düşük skor = iyi (hem hedefe yakın hem SLL düşük).
    
    Kısıt ihlali olan satırlar elenir (fiziksel olarak geçersiz).
    """
    valid = df[df["violations_mean"] < 0.5].copy()
    if len(valid) == 0:
        print("  ⚠ Tüm sonuçlarda kısıt ihlali var, en düşük ihlalli satır seçiliyor.")
        valid = df.copy()

    # Normalize et (0-1 aralığına)
    err_norm = (valid["main_lobe_error_mean"] - valid["main_lobe_error_mean"].min()) / \
               (valid["main_lobe_error_mean"].max() - valid["main_lobe_error_mean"].min() + 1e-9)
    sll_norm = (valid["sll_db_mean"] - valid["sll_db_mean"].min()) / \
               (valid["sll_db_mean"].max() - valid["sll_db_mean"].min() + 1e-9)

    valid["score"] = error_weight * err_norm + sll_weight * sll_norm
    best_row = valid.loc[valid["score"].idxmin()]

    return best_row.to_dict()
