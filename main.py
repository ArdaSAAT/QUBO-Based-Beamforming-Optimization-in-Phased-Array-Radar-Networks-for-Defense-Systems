"""
QUBO BeamForming — Ana Pipeline
Aşama 1-2: Parametre Tanımı + Sentetik Veri Üretimi

Çalıştır:
    python main.py
"""

import numpy as np
import os
from src.array_factor import (
    ArrayConfig,
    calculate_array_factor,
    beam_metrics,
    broadside_config,
    steering_phases,
)
from src.evaluation import plot_scenario, plot_all_scenarios, print_metrics

# ─────────────────────────────────────────────────────────────
# ORTAK AYARLAR
# ─────────────────────────────────────────────────────────────

THETA_RANGE = np.linspace(-90, 90, 181)   # 1° adım, 181 nokta
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  QUBO BeamForming | Faz Dizili Radar Anten Simülasyonu")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SENARYO TANIMLARI (Proje planından aynen)
# ─────────────────────────────────────────────────────────────

scenarios_cfg = [
    # ① Senaryo 1: 8 eleman, θ=0° — Başlangıç / Doğrulama
    ArrayConfig(N=8,  b_phase=3, b_amp=2, theta_target=0.0,
                theta_nulls=[], sll_target=-20.0),

    # ② Senaryo 2: 16 eleman, θ=30° — Temel karşılaştırma
    ArrayConfig(N=16, b_phase=3, b_amp=2, theta_target=30.0,
                theta_nulls=[], sll_target=-20.0),

    # ③ Senaryo 3: 32 eleman, θ=-20° — Ölçeklendirme testi
    ArrayConfig(N=32, b_phase=3, b_amp=2, theta_target=-20.0,
                theta_nulls=[], sll_target=-25.0),

    # ④ Senaryo 4: 16 eleman, çoklu null — Jammer bastırma
    ArrayConfig(N=16, b_phase=4, b_amp=2, theta_target=0.0,
                theta_nulls=[-40.0, 40.0], sll_target=-25.0),

    # ⑤ Senaryo 5: 64 eleman, θ=45° — Büyük ölçek (referans)
    ArrayConfig(N=64, b_phase=3, b_amp=2, theta_target=45.0,
                theta_nulls=[], sll_target=-25.0),
]

scenario_labels = [
    "Senaryo 1 | N=8  | θ=0°   | Broadside Doğrulama",
    "Senaryo 2 | N=16 | θ=30°  | Beam Steering",
    "Senaryo 3 | N=32 | θ=-20° | Ölçeklendirme",
    "Senaryo 4 | N=16 | θ=0°   | Çift Null (Jammer Bastırma)",
    "Senaryo 5 | N=64 | θ=45°  | Büyük Ölçek Referans",
]


# ─────────────────────────────────────────────────────────────
# HER SENARYO İÇİN: 2 Referans Konfigürasyon
#   A) Broadside (A=1, ψ=0°)     — QUBO öncesi taban çizgisi
#   B) Klasik Steering (analitik) — optimum faz kaydırma
# ─────────────────────────────────────────────────────────────

all_scenario_data = []  # özet panel için

for i, (cfg, label) in enumerate(zip(scenarios_cfg, scenario_labels), start=1):
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"  {cfg}")
    print(f"{'═'*60}")

    # A) Broadside referansı
    amp_broad, pha_broad = broadside_config(cfg)

    # B) Klasik faz kaydırma (sürekli, optimal)
    amp_steer = np.ones(cfg.N)
    pha_steer = steering_phases(cfg)

    # Metrikleri yazdır
    print_metrics("Broadside (referans)", cfg, amp_broad, pha_broad, THETA_RANGE)
    print_metrics("Klasik Steering (analitik)", cfg, amp_steer, pha_steer, THETA_RANGE)

    # Görselleştir
    results = {
        "Broadside":      (amp_broad, pha_broad),
        "Klasik Steering": (amp_steer, pha_steer),
    }
    plot_scenario(
        config=cfg,
        thetas_deg=THETA_RANGE,
        results=results,
        title=label,
        save_path=f"{OUTPUT_DIR}/senaryo_{i}.png",
    )

    # Özet panel için kaydet (klasik steering konfigürasyonu)
    all_scenario_data.append({
        "config":     cfg,
        "amplitudes": amp_steer,
        "phases_deg": pha_steer,
        "label":      f"Senaryo {i}",
    })


# ─────────────────────────────────────────────────────────────
# ÖZET PANEL: Tüm senaryolar tek görüntüde
# ─────────────────────────────────────────────────────────────

print("\n\nÖzet panel oluşturuluyor...")
plot_all_scenarios(
    scenarios=all_scenario_data,
    thetas_deg=THETA_RANGE,
    save_path=f"{OUTPUT_DIR}/tum_senaryolar.png",
)

print(f"\n{'='*60}")
print(f"  ✓ Aşama 1-2 tamamlandı.")
print(f"  Sonraki adım → Aşama 3: QUBO formülasyonu (qubo_formulation.py)")
print(f"{'='*60}\n")
