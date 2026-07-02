"""
Aşama 1-2: Parametre Tanımı ve Sentetik Veri Üretimi
Uniform Linear Array (ULA) için Array Factor fizik motoru.
"""

import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# VERİ YAPISI: Tüm anten parametreleri tek bir objede
# ─────────────────────────────────────────────────────────────

@dataclass
class ArrayConfig:
    """
    Faz dizili anten konfigürasyonu.
    
    Parametreler:
        N          : Eleman sayısı
        b_phase    : Faz çözünürlüğü (bit) → 3-bit = 8 seviye = 0°,45°,...,315°
        b_amp      : Genlik çözünürlüğü (bit) → 2-bit = 4 seviye = 0, 0.33, 0.67, 1.0
        theta_target: Hedef huzme açısı (derece)
        theta_nulls : Null oluşturulacak açılar listesi (derece) — jammer bastırma
        sll_target  : Hedef yan lob seviyesi (dB, negatif)
        d_over_lambda: Eleman aralığı / dalga boyu (varsayılan 0.5)
    """
    N: int = 8
    b_phase: int = 3
    b_amp: int = 2
    theta_target: float = 0.0
    theta_nulls: list = field(default_factory=list)
    sll_target: float = -20.0
    d_over_lambda: float = 0.5

    # Türetilen özellikler
    @property
    def n_phase_levels(self):
        """Kaç farklı faz değeri var? (3-bit → 8)"""
        return 2 ** self.b_phase

    @property
    def phase_levels_deg(self):
        """Mevcut faz seviyeleri (derece)"""
        return np.linspace(0, 360, self.n_phase_levels, endpoint=False)

    @property
    def n_amp_levels(self):
        """Kaç farklı genlik değeri var? (2-bit → 4)"""
        return 2 ** self.b_amp

    @property
    def amp_levels(self):
        """Mevcut genlik seviyeleri (0.0 → 1.0)"""
        return np.linspace(0, 1, self.n_amp_levels)

    @property
    def n_vars(self):
        """Toplam binary değişken sayısı (QUBO boyutu)"""
        return self.N * (self.b_phase + self.b_amp)

    def __str__(self):
        return (
            f"ArrayConfig | N={self.N} eleman | "
            f"{self.b_phase}-bit faz ({self.n_phase_levels} seviye) | "
            f"{self.b_amp}-bit genlik ({self.n_amp_levels} seviye) | "
            f"θ_hedef={self.theta_target}° | "
            f"QUBO boyutu={self.n_vars} binary değişken"
        )


# ─────────────────────────────────────────────────────────────
# FİZİK MOTORU: Array Factor Hesaplama
# ─────────────────────────────────────────────────────────────

def calculate_array_factor(config: ArrayConfig,
                           amplitudes: np.ndarray,
                           phases_deg: np.ndarray,
                           thetas_deg: np.ndarray) -> np.ndarray:
    """
    Verilen genlik ve faz değerleri için Array Factor hesaplar.
    
    Formül:
        AF(θ) = Σ Aₙ · exp(j·ψₙ) · exp(j·n·k·d·sin(θ))
        k·d = 2π · (d/λ) = π  (d=λ/2 için)
    
    Dönüş:
        AF_db: Normalize edilmiş dB cinsinden Array Factor (shape: len(thetas_deg),)
    """
    thetas_rad = np.deg2rad(thetas_deg)
    phases_rad = np.deg2rad(phases_deg)
    kd = 2 * np.pi * config.d_over_lambda  # k·d = π (d=λ/2 için)

    n_array = np.arange(config.N)

    # Steering matrisi: (N × M) — M = açı sayısı
    steering_matrix = np.exp(1j * kd * np.outer(n_array, np.sin(thetas_rad)))

    # Kompleks ağırlık vektörü: Aₙ · exp(j·ψₙ)
    weights = amplitudes * np.exp(1j * phases_rad)

    # Vektörel çarpım → AF (shape: M,)
    AF_linear = np.dot(weights, steering_matrix)

    # Normalize et ve dB'e çevir
    AF_abs = np.abs(AF_linear)
    AF_normalized = AF_abs / (np.max(AF_abs) + 1e-12)
    AF_db = 20 * np.log10(AF_normalized + 1e-10)

    return AF_db


def calculate_af_complex(config: ArrayConfig,
                         amplitudes: np.ndarray,
                         phases_deg: np.ndarray,
                         thetas_deg: np.ndarray) -> np.ndarray:
    """
    Kompleks AF değerini döndürür (metrik hesaplama için gerekli).
    """
    thetas_rad = np.deg2rad(thetas_deg)
    phases_rad = np.deg2rad(phases_deg)
    kd = 2 * np.pi * config.d_over_lambda

    n_array = np.arange(config.N)
    steering_matrix = np.exp(1j * kd * np.outer(n_array, np.sin(thetas_rad)))
    weights = amplitudes * np.exp(1j * phases_rad)

    return np.dot(weights, steering_matrix)


# ─────────────────────────────────────────────────────────────
# METRİK HESAPLAMA: SLL, HPBW, Null Depth
# ─────────────────────────────────────────────────────────────

def beam_metrics(config: ArrayConfig,
                 amplitudes: np.ndarray,
                 phases_deg: np.ndarray,
                 thetas_deg: np.ndarray) -> dict:
    """
    Verilen konfigürasyon için tüm beam performans metriklerini hesaplar.
    
    Dönüş (dict):
        sll_db       : Side Lobe Level (dB) — düşük = iyi
        hpbw_deg     : Half Power Beam Width (derece) — dar = iyi
        main_lobe_deg: Ana lobun gerçek yön açısı
        null_depths  : Her null açısı için zayıflatma (dB)
        af_db        : Tam radyasyon örüntüsü (görselleştirme için)
    """
    AF_complex = calculate_af_complex(config, amplitudes, phases_deg, thetas_deg)
    AF_abs = np.abs(AF_complex)
    AF_max = np.max(AF_abs)

    # Ana lob: maksimum genlik noktası
    main_idx = np.argmax(AF_abs)
    main_lobe_deg = thetas_deg[main_idx]

    # ── SLL Hesabı ──────────────────────────────────────────
    # Ana lob bölgesi dışındaki en yüksek değeri bul
    # Ana lobun yarı güç (-3dB) genişliği bulunarak dışarısı "yan lob" sayılır
    half_power = AF_max / np.sqrt(2)

    # Ana lob bölgesini belirle (main_idx'ten iki yana doğru genişle)
    left = main_idx
    while left > 0 and AF_abs[left] > half_power:
        left -= 1
    right = main_idx
    while right < len(AF_abs) - 1 and AF_abs[right] > half_power:
        right += 1

    # Yan lob maskesi
    sidelobe_mask = np.ones(len(thetas_deg), dtype=bool)
    sidelobe_mask[left:right+1] = False

    if np.any(sidelobe_mask):
        sll_linear = np.max(AF_abs[sidelobe_mask]) / AF_max
        sll_db = 20 * np.log10(sll_linear + 1e-12)
    else:
        sll_db = -60.0  # Yan lob yok (çok iyi)

    # ── HPBW Hesabı ─────────────────────────────────────────
    hpbw_deg = thetas_deg[right] - thetas_deg[left]

    # ── Null Depth Hesabı ────────────────────────────────────
    null_depths = {}
    for theta_null in config.theta_nulls:
        null_idx = np.argmin(np.abs(thetas_deg - theta_null))
        null_linear = AF_abs[null_idx] / (AF_max + 1e-12)
        null_depths[theta_null] = 20 * np.log10(null_linear + 1e-12)

    # ── Normalize dB ─────────────────────────────────────────
    AF_normalized = AF_abs / (AF_max + 1e-12)
    AF_db = 20 * np.log10(AF_normalized + 1e-10)

    return {
        "sll_db": sll_db,
        "hpbw_deg": hpbw_deg,
        "main_lobe_deg": main_lobe_deg,
        "null_depths": null_depths,
        "af_db": AF_db,
        "af_abs": AF_abs,
    }


# ─────────────────────────────────────────────────────────────
# BROADSIDE REFERANS: Başlangıç / Doğrulama Konfigürasyonu
# ─────────────────────────────────────────────────────────────

def broadside_config(config: ArrayConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Tüm elemanlar A=1, ψ=0° → Broadside huzme (θ=0°).
    Senaryo 1 doğrulama referansı olarak kullanılır.
    """
    amplitudes = np.ones(config.N)
    phases_deg = np.zeros(config.N)
    return amplitudes, phases_deg


def steering_phases(config: ArrayConfig) -> np.ndarray:
    """
    θ_target yönüne klasik faz kaydırma (uniform amplitude).
    Formül: ψₙ = -n · k·d · sin(θ_target)
    QUBO çözücülerin karşılaştırma referansı.
    """
    kd = 2 * np.pi * config.d_over_lambda
    n_array = np.arange(config.N)
    phases_rad = -n_array * kd * np.sin(np.deg2rad(config.theta_target))
    return np.rad2deg(phases_rad) % 360
