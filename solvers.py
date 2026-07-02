"""
Aşama 4: Optimizasyon Çözücüleri
Şu an: Simulated Annealing (SA) — dwave.samplers
Sırada: QAOA (Qiskit), D-Wave QPU
"""

import numpy as np
import time
from dataclasses import dataclass
from dwave.samplers import SimulatedAnnealingSampler
from src.array_factor import ArrayConfig, beam_metrics
from src.qubo_formulation import build_qubo_onehot, decode_solution


# ─────────────────────────────────────────────────────────────
# SONUÇ YAPISI
# ─────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    """Bir çözücü çalışmasının tam sonucu."""
    method: str                  # "SA", "QAOA", "DWave"
    amplitudes: np.ndarray
    phases_deg: np.ndarray
    sll_db: float
    hpbw_deg: float
    main_lobe_deg: float
    null_depths: dict
    best_energy: float
    solve_time_sec: float
    n_violations: int
    lambda_sll: float
    lambda_null: float

    def __str__(self):
        lines = [
            f"\n{'─'*55}",
            f"  Yöntem       : {self.method}",
            f"  Ana Lob      : {self.main_lobe_deg:+.1f}°",
            f"  SLL          : {self.sll_db:.2f} dB",
            f"  HPBW         : {self.hpbw_deg:.2f}°",
            f"  Enerji       : {self.best_energy:.3f}",
            f"  Süre         : {self.solve_time_sec:.2f} sn",
            f"  Kısıt ihlali : {self.n_violations}",
        ]
        for theta, depth in self.null_depths.items():
            lines.append(f"  Null @ {theta:+.0f}°  : {depth:.2f} dB")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# SİMULATED ANNEALING ÇÖZÜCÜsü
# ─────────────────────────────────────────────────────────────

def solve_sa(config: ArrayConfig,
             thetas_deg: np.ndarray,
             lambda_sll: float = 0.1,
             lambda_null: float = 2.0,
             num_reads: int = 500,
             num_sweeps: int = 2000,
             verbose: bool = True) -> SolverResult:
    """
    Simulated Annealing ile QUBO çözer.

    num_reads  : Kaç bağımsız SA çalışması yapılsın (en iyisi seçilir)
    num_sweeps : Her çalışmada kaç adım atılsın (daha fazla = daha iyi ama yavaş)
    """
    if verbose:
        print(f"\n  [SA] QUBO kuruluyor... (N={config.N}, λ_sll={lambda_sll}, λ_null={lambda_null})")

    # QUBO kur
    t0 = time.time()
    H, variables = build_qubo_onehot(
        config,
        lambda_sll=lambda_sll,
        lambda_null=lambda_null,
    )
    qubo, offset = H.compile().to_qubo()

    if verbose:
        print(f"  [SA] Q matrisi: {len(qubo)} terim | {num_reads} okuma × {num_sweeps} adım")

    # SA çalıştır
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(
        qubo,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
    )
    solve_time = time.time() - t0

    # En iyi çözümü decode et
    best = sampleset.first
    amps, phases, n_viol = decode_solution(best.sample, variables)

    # Metrikleri hesapla
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
        solve_time_sec=solve_time,
        n_violations=n_viol,
        lambda_sll=lambda_sll,
        lambda_null=lambda_null,
    )

    if verbose:
        print(result)

    return result


# ─────────────────────────────────────────────────────────────
# KARŞILAŞTIRMA: Klasik vs SA
# ─────────────────────────────────────────────────────────────

def compare_results(classical_amps: np.ndarray,
                    classical_phases: np.ndarray,
                    sa_result: SolverResult,
                    config: ArrayConfig,
                    thetas_deg: np.ndarray) -> dict:
    """
    Klasik steering ile SA sonucunu karşılaştırır,
    kazanım/kayıp tablosu üretir.
    """
    m_c = beam_metrics(config, classical_amps, classical_phases, thetas_deg)

    delta_sll  = sa_result.sll_db - m_c["sll_db"]
    delta_hpbw = sa_result.hpbw_deg - m_c["hpbw_deg"]
    delta_lobe = abs(sa_result.main_lobe_deg - config.theta_target) - \
                 abs(m_c["main_lobe_deg"] - config.theta_target)

    print(f"\n{'═'*55}")
    print(f"  KARŞILAŞTIRMA: Klasik Steering vs SA")
    print(f"{'═'*55}")
    print(f"  {'Metrik':<20} {'Klasik':>10} {'SA':>10} {'Δ':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Ana Lob Sapması':<20} {abs(m_c['main_lobe_deg']-config.theta_target):>9.1f}° "
          f"{abs(sa_result.main_lobe_deg-config.theta_target):>9.1f}° "
          f"{delta_lobe:>+9.1f}°")
    print(f"  {'SLL':<20} {m_c['sll_db']:>9.2f}  {sa_result.sll_db:>9.2f}  {delta_sll:>+9.2f} dB")
    print(f"  {'HPBW':<20} {m_c['hpbw_deg']:>9.1f}° {sa_result.hpbw_deg:>9.1f}° {delta_hpbw:>+9.1f}°")
    print(f"{'═'*55}")

    # SLL için negatif delta = SA daha iyi bastırıyor
    if delta_sll < -1.0:
        print("  ✓ SA, yan lobları klasikten belirgin şekilde daha iyi bastırdı.")
    elif delta_sll > 1.0:
        print("  ✗ Klasik yöntem SLL'de daha iyi — λ_sll ayarına bakılmalı.")
    else:
        print("  ~ SLL farkı 1 dB'den az — iki yöntem bu metrikte benzer.")

    return {
        "classical": m_c,
        "sa": sa_result,
        "delta_sll": delta_sll,
        "delta_hpbw": delta_hpbw,
    }
