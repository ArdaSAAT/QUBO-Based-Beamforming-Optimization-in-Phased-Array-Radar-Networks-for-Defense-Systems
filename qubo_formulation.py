"""
Aşama 5: Değerlendirme ve Görselleştirme
Radyasyon örüntüleri, metrik tabloları, karşılaştırma grafikleri.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.array_factor import ArrayConfig, beam_metrics


# ─────────────────────────────────────────────────────────────
# TEMA
# ─────────────────────────────────────────────────────────────

COLORS = {
    "broadside":  "#00BFFF",   # referans — açık mavi
    "classical":  "#32CD32",   # klasik steering — yeşil
    "sa":         "#FFA500",   # simulated annealing — turuncu
    "qaoa":       "#FF69B4",   # QAOA — pembe
    "dwave":      "#9370DB",   # D-Wave — mor
    "target":     "#FF4444",   # hedef çizgisi — kırmızı
    "null":       "#FFFF00",   # null çizgisi — sarı
}


# ─────────────────────────────────────────────────────────────
# TEK SENARYO: Kutupsal + Kartezyen çift panel
# ─────────────────────────────────────────────────────────────

def plot_scenario(config: ArrayConfig,
                  thetas_deg: np.ndarray,
                  results: dict,
                  title: str = "",
                  save_path: str = None):
    """
    Tek bir senaryo için iki panelli görselleştirme:
      Sol  → Kutupsal radyasyon örüntüsü
      Sağ  → Kartezyen dB grafiği + metrik tablosu
    
    results = {
        "label": (amplitudes, phases_deg),
        ...
    }
    """
    fig = plt.figure(figsize=(16, 7), facecolor="#0D1117")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_polar = fig.add_subplot(gs[0], projection="polar")
    ax_cart  = fig.add_subplot(gs[1])

    # Stil
    for ax in [ax_cart]:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    ax_polar.set_facecolor("#161B22")
    ax_polar.tick_params(colors="white")

    metric_rows = []

    for i, (label, (amps, phases)) in enumerate(results.items()):
        color = list(COLORS.values())[i % len(COLORS)]

        m = beam_metrics(config, amps, phases, thetas_deg)
        af_db = m["af_db"]

        # Kutupsal: sadece [-90°, 90°] göster, dB'i 0'dan başlat
        af_plot = np.clip(af_db, -40, 0)
        af_shifted = af_plot + 40  # 0-40 aralığına taşı (kutupsal için)

        ax_polar.plot(np.deg2rad(thetas_deg), af_shifted,
                      color=color, linewidth=1.8, label=label, alpha=0.9)

        # Kartezyen
        ax_cart.plot(thetas_deg, af_db,
                     color=color, linewidth=1.8, label=label, alpha=0.9)

        metric_rows.append({
            "Yöntem": label,
            "SLL (dB)": f"{m['sll_db']:.1f}",
            "HPBW (°)": f"{m['hpbw_deg']:.1f}",
            "Ana Lob (°)": f"{m['main_lobe_deg']:.1f}",
        })

    # Hedef ve null çizgileri
    ax_cart.axvline(config.theta_target, color=COLORS["target"],
                    linestyle="--", linewidth=1.2, alpha=0.7, label=f"Hedef θ={config.theta_target}°")
    for tn in config.theta_nulls:
        ax_cart.axvline(tn, color=COLORS["null"],
                        linestyle=":", linewidth=1.2, alpha=0.7, label=f"Null θ={tn}°")

    # SLL hedef çizgisi
    ax_cart.axhline(config.sll_target, color="#FF4444",
                    linestyle="-.", linewidth=0.8, alpha=0.5, label=f"SLL hedef {config.sll_target} dB")

    # Kartezyen eksen ayarları
    ax_cart.set_xlim(-90, 90)
    ax_cart.set_ylim(-45, 2)
    ax_cart.set_xlabel("Açı θ (derece)", color="white", fontsize=11)
    ax_cart.set_ylabel("Normalize Genlik (dB)", color="white", fontsize=11)
    ax_cart.set_xticks(range(-90, 91, 15))
    ax_cart.set_yticks(range(-40, 5, 5))
    ax_cart.grid(color="#30363D", linestyle="--", linewidth=0.5)
    ax_cart.legend(loc="lower right", facecolor="#161B22",
                   labelcolor="white", fontsize=8, framealpha=0.8)

    # Kutupsal eksen ayarları
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_thetamin(-90)
    ax_polar.set_thetamax(90)
    ax_polar.set_ylim(0, 42)
    ax_polar.set_yticks([0, 10, 20, 30, 40])
    ax_polar.set_yticklabels(["-40", "-30", "-20", "-10", "0"], color="gray", fontsize=7)
    ax_polar.grid(color="#30363D", linewidth=0.5)
    ax_polar.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15),
                    facecolor="#161B22", labelcolor="white", fontsize=8)

    # Başlık
    main_title = title or f"N={config.N} | θ_hedef={config.theta_target}° | {config.b_phase}-bit faz"
    fig.suptitle(main_title, color="white", fontsize=14, fontweight="bold", y=1.01)

    # Metrik tablosu (alt)
    _add_metric_table(fig, metric_rows)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✓ Kaydedildi: {save_path}")
    plt.show()


def _add_metric_table(fig, rows: list[dict]):
    """Grafiğin altına metrik tablosu ekler."""
    if not rows:
        return
    cols = list(rows[0].keys())
    cell_text = [[r[c] for c in cols] for r in rows]
    colors_col = [list(COLORS.values())[i % len(COLORS)] for i in range(len(rows))]

    ax_table = fig.add_axes([0.05, -0.18, 0.9, 0.15])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=cell_text,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#161B22" if row > 0 else "#21262D")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#30363D")
        if row > 0 and col == 0:
            cell.set_facecolor(colors_col[row - 1] + "33")  # renk tonu


# ─────────────────────────────────────────────────────────────
# TÜM SENARYOLAR: Özet karşılaştırma paneli
# ─────────────────────────────────────────────────────────────

def plot_all_scenarios(scenarios: list[dict], thetas_deg: np.ndarray,
                       save_path: str = None):
    """
    5 senaryonun hepsini 2×3 grid'de gösterir.
    
    scenarios = [
        {
            "config": ArrayConfig(...),
            "amplitudes": np.array,
            "phases_deg": np.array,
            "label": "Senaryo 1 - Broadside"
        },
        ...
    ]
    """
    n = len(scenarios)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows),
                              facecolor="#0D1117",
                              subplot_kw={"projection": "polar"})
    axes = np.array(axes).flatten()

    for i, (sc, ax) in enumerate(zip(scenarios, axes)):
        config = sc["config"]
        amps = sc["amplitudes"]
        phases = sc["phases_deg"]
        label = sc.get("label", f"Senaryo {i+1}")

        m = beam_metrics(config, amps, phases, thetas_deg)
        af_plot = np.clip(m["af_db"], -40, 0) + 40

        color = list(COLORS.values())[i % len(COLORS)]
        ax.plot(np.deg2rad(thetas_deg), af_plot, color=color, linewidth=1.8)
        ax.set_facecolor("#161B22")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_ylim(0, 42)
        ax.set_yticks([0, 20, 40])
        ax.set_yticklabels(["-40", "-20", "0"], color="gray", fontsize=7)
        ax.grid(color="#30363D", linewidth=0.5)
        ax.set_title(
            f"{label}\nSLL={m['sll_db']:.1f} dB | HPBW={m['hpbw_deg']:.1f}°",
            color="white", fontsize=9, pad=10
        )

    # Kullanılmayan eksenleri gizle
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("QUBO BeamForming — Tüm Senaryolar",
                 color="white", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✓ Özet panel kaydedildi: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# METRİK RAPORU: Konsola yazdır
# ─────────────────────────────────────────────────────────────

def print_metrics(label: str, config: ArrayConfig,
                  amplitudes: np.ndarray, phases_deg: np.ndarray,
                  thetas_deg: np.ndarray):
    """Metrikleri okunabilir formatta konsola yazdırır."""
    m = beam_metrics(config, amplitudes, phases_deg, thetas_deg)

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Ana Lob Yönü : {m['main_lobe_deg']:+.1f}°  (hedef: {config.theta_target:+.1f}°)")
    print(f"  SLL          : {m['sll_db']:.2f} dB  (hedef: < {config.sll_target} dB)  "
          f"{'✓' if m['sll_db'] < config.sll_target else '✗'}")
    print(f"  HPBW         : {m['hpbw_deg']:.2f}°")

    for theta_null, depth in m["null_depths"].items():
        print(f"  Null @ {theta_null:+.0f}°  : {depth:.2f} dB  "
              f"{'✓' if depth < -30 else '✗'}")
    print(f"{'─'*55}")

    return m
