"""
Aşama 3: QUBO Formülasyonu
Faz dizili anten optimizasyon problemini QUBO (Quadratic Unconstrained
Binary Optimization) formuna dönüştürür.

Matematiksel arka plan:
─────────────────────────
Her eleman n için:
  - Faz ψₙ, b_phase bitlik bir binary kodla temsil edilir → 2^b_phase seviye
  - Genlik Aₙ, b_amp bitlik bir binary kodla temsil edilir → 2^b_amp seviye

Maliyet fonksiyonu üç bileşenden oluşur:
  1. Ana lob kazancı   : Hedef açıda |AF|² büyük olsun (negatif işaretle minimize)
  2. Yan lob enerjisi   : Yan lob bölgesinde |AF|² küçük olsun (λ1 ağırlıklı)
  3. Null derinliği     : Null açılarında |AF|² ≈ 0 olsun (λ2 ağırlıklı)

  Cost = -Gain(θ_target) + λ1 · Σ|AF(θ_sidelobe)|² + λ2 · Σ|AF(θ_null)|²
"""

import numpy as np
from pyqubo import Array, Constraint
from src.array_factor import ArrayConfig


# ─────────────────────────────────────────────────────────────
# BİNARY → FİZİKSEL DEĞER DÖNÜŞÜMÜ
# ─────────────────────────────────────────────────────────────

def binary_to_phase_expr(bits, b_phase):
    """
    b_phase bitlik binary vektörü 0-360° aralığında faza çevirir.
    İkili kodlama: ψ = (Σ bᵢ·2ⁱ) / (2^b_phase) · 360°
    
    bits: pyqubo Binary değişken listesi (uzunluk b_phase)
    Dönüş: sembolik pyqubo ifadesi (derece)
    """
    n_levels = 2 ** b_phase
    weighted_sum = sum(bits[i] * (2 ** i) for i in range(b_phase))
    return weighted_sum * (360.0 / n_levels)


def binary_to_amp_expr(bits, b_amp):
    """
    b_amp bitlik binary vektörü 0-1 aralığında genliğe çevirir.
    A = (Σ bᵢ·2ⁱ) / (2^b_amp - 1)
    """
    n_levels = 2 ** b_amp
    max_val = n_levels - 1
    weighted_sum = sum(bits[i] * (2 ** i) for i in range(b_amp))
    return weighted_sum * (1.0 / max_val) if max_val > 0 else weighted_sum


def binary_to_phase_numeric(bit_values, b_phase):
    """Sayısal versiyon (çözüm sonrası decode için)."""
    n_levels = 2 ** b_phase
    weighted_sum = sum(bit_values[i] * (2 ** i) for i in range(b_phase))
    return weighted_sum * (360.0 / n_levels)


def binary_to_amp_numeric(bit_values, b_amp):
    """Sayısal versiyon (çözüm sonrası decode için)."""
    n_levels = 2 ** b_amp
    max_val = n_levels - 1
    weighted_sum = sum(bit_values[i] * (2 ** i) for i in range(b_amp))
    return weighted_sum * (1.0 / max_val) if max_val > 0 else weighted_sum


# ─────────────────────────────────────────────────────────────
# ÖNEMLİ NOT — DOĞRUSAL OLMAYAN TERİMLER
# ─────────────────────────────────────────────────────────────
# Array Factor'ün içinde cos(ψ) ve sin(ψ) gibi trigonometrik terimler var.
# Bunlar binary değişkenlerin DOĞRUSAL OLMAYAN fonksiyonu, dolayısıyla
# doğrudan QUBO'ya (kuadratik forma) sığmaz.
#
# Çözüm: Faz açısını bit kombinasyonlarının her biri için ÖNCEDEN hesaplayıp
# (lookup table), cos/sin değerlerini sabit katsayı olarak QUBO'ya gömüyoruz.
# Bu "one-hot kodlama" yaklaşımına yakın ama bit-ağırlıklı (binary-weighted)
# kodlama kullandığımız için her bit kombinasyonu ayrı bir terim üretir.
#
# Bu yüzden gerçek QUBO kurulumunda, ikili-ağırlıklı (binary positional)
# kodlama yerine, her eleman için "hangi faz seviyesi aktif" sorusunu
# ONE-HOT kodlama ile soracağız: bu suretle cos/sin sabit katsayı olur
# ve QUBO kuadratik kalır.
# ─────────────────────────────────────────────────────────────


def build_qubo_onehot(config: ArrayConfig,
                       theta_target: float = None,
                       theta_sidelobe_region: np.ndarray = None,
                       theta_nulls: list = None,
                       lambda_sll: float = 1.0,
                       lambda_null: float = 5.0):
    """
    One-hot kodlama ile QUBO kurar (doğru ve standart yaklaşım).
    
    Her eleman n için:
      - b_phase bit YERİNE, n_phase_levels adet one-hot binary değişken
        (sadece biri 1, gerisi 0 → "hangi faz seviyesi seçili")
      - Genlik için de aynı mantık (n_amp_levels one-hot)
    
    Bu yaklaşım QUBO'yu kuadratik tutar çünkü cos(ψ_k) sabit sayıdır,
    binary değişkenle çarpılır → doğrusal/kuadratik terim.
    
    Dönüş:
        H (pyqubo Model bağlanmamış ifade), variables (dict: değişken referansları)
    """
    N = config.N
    phase_levels = config.phase_levels_deg          # örn: [0,45,90,...,315]
    amp_levels = config.amp_levels                   # örn: [0, .33, .67, 1.0]
    n_phase = len(phase_levels)
    n_amp = len(amp_levels)

    if theta_target is None:
        theta_target = config.theta_target
    if theta_nulls is None:
        theta_nulls = config.theta_nulls

    # ── Binary değişkenler: x[n][k] = eleman n, faz seviyesi k seçili mi? ──
    x_phase = Array.create("x_phase", shape=(N, n_phase), vartype="BINARY")
    x_amp   = Array.create("x_amp",   shape=(N, n_amp),   vartype="BINARY")

    kd = 2 * np.pi * config.d_over_lambda

    # ── Her eleman için kompleks ağırlığın reel/imajiner kısmı ──
    # wₙ = Aₙ · exp(jψₙ) = Aₙ·cos(ψₙ) + j·Aₙ·sin(ψₙ)
    # One-hot kodlamada: Aₙ = Σₖ amp_levels[k]·x_amp[n,k]
    #                    cos(ψₙ) = Σₖ cos(phase_levels[k])·x_phase[n,k]  (yaklaşık ayrıştırma)
    #
    # NOT: Aₙ ve ψₙ aynı anda one-hot olduğundan wₙ_re ve wₙ_im
    # iki one-hot değişkenin ÇARPIMI olur → kuadratik terim (QUBO'ya uygun).

    def amplitude_expr(n):
        return sum(amp_levels[k] * x_amp[n, k] for k in range(n_amp))

    def cos_phase_expr(n):
        return sum(np.cos(np.deg2rad(phase_levels[k])) * x_phase[n, k] for k in range(n_phase))

    def sin_phase_expr(n):
        return sum(np.sin(np.deg2rad(phase_levels[k])) * x_phase[n, k] for k in range(n_phase))

    def af_real_imag(theta_deg):
        """Belirli bir açıda AF'nin reel ve imajiner kısmını sembolik döndürür."""
        theta_rad = np.deg2rad(theta_deg)
        re_total, im_total = 0, 0
        for n in range(N):
            phase_n = n * kd * np.sin(theta_rad)
            cos_n, sin_n = np.cos(phase_n), np.sin(phase_n)
            amp_n = amplitude_expr(n)
            cosw_n = cos_phase_expr(n)
            sinw_n = sin_phase_expr(n)

            # wₙ · exp(j·n·k·d·sinθ) reel/imajiner ayrıştırma
            # wₙ_re = Aₙ·cosψₙ,  wₙ_im = Aₙ·sinψₙ  -- bunlar 2 one-hot'un çarpımı (kuadratik)
            w_re = amp_n * cosw_n
            w_im = amp_n * sinw_n

            re_total += w_re * cos_n - w_im * sin_n
            im_total += w_re * sin_n + w_im * cos_n
        return re_total, im_total

    # ── Hedef açıda güç (maksimize edilecek → negatif ekleyerek minimize) ──
    re_t, im_t = af_real_imag(theta_target)
    power_target = re_t ** 2 + im_t ** 2

    # ── Yan lob bölgesi enerjisi (minimize edilecek) ──
    if theta_sidelobe_region is None:
        # Varsayılan: hedeften ±15° dışındaki [-90,90] aralığı, 10° adımlarla örnekle
        all_thetas = np.arange(-90, 91, 10)
        theta_sidelobe_region = all_thetas[np.abs(all_thetas - theta_target) > 15]

    # NORMALIZASYON: Sidelobe bölgesi onlarca nokta içerebilir; her nokta
    # power_target ile aynı mertebede bir terim ekliyor. Normalize etmezsek
    # toplam sidelobe cezası tek bir ana-lob teriminden N_nokta kat ağır basar
    # ve optimizer ana lobu feda eder. Bu yüzden NOKTA SAYISINA BÖLEREK
    # "ortalama yan lob gücü" haline getiriyoruz — artık lambda_sll, ana lob
    # gücüne göre nispi bir ağırlık anlamına geliyor (örn. 0.1 = %10 önem).
    sidelobe_energy = 0
    for theta in theta_sidelobe_region:
        re_s, im_s = af_real_imag(theta)
        sidelobe_energy += re_s ** 2 + im_s ** 2
    if len(theta_sidelobe_region) > 0:
        sidelobe_energy = sidelobe_energy / len(theta_sidelobe_region)

    # ── Null bölgesi enerjisi (ağır cezalandırılacak) ──
    # Null noktaları genelde az sayıda (1-3) olduğundan ana lob terimiyle
    # zaten aynı mertebede kalıyor, yine de tutarlılık için normalize ediyoruz.
    null_energy = 0
    for theta in theta_nulls:
        re_n, im_n = af_real_imag(theta)
        null_energy += re_n ** 2 + im_n ** 2
    if len(theta_nulls) > 0:
        null_energy = null_energy / len(theta_nulls)

    # ── Toplam Hamiltonyen ──
    H = -power_target + lambda_sll * sidelobe_energy + lambda_null * null_energy

    # ── One-hot kısıtı: her eleman için TAM OLARAK bir faz / bir genlik seçilmeli ──
    # Bu olmadan x_phase[n,:] hepsi 0 ya da hepsi 1 olabilir → fiziksel olarak anlamsız
    onehot_penalty = 0
    constraint_strength = 10.0 * (abs(float(H.compile().to_qubo()[1])) if False else 1.0)
    # Basit ve sağlam sabit ceza katsayısı (deneyle ayarlanabilir)
    PENALTY = 50.0

    for n in range(N):
        onehot_penalty += PENALTY * (sum(x_phase[n, k] for k in range(n_phase)) - 1) ** 2
        onehot_penalty += PENALTY * (sum(x_amp[n, k] for k in range(n_amp)) - 1) ** 2

    H_total = H + onehot_penalty

    variables = {
        "x_phase": x_phase,
        "x_amp": x_amp,
        "n_phase": n_phase,
        "n_amp": n_amp,
        "phase_levels": phase_levels,
        "amp_levels": amp_levels,
        "N": N,
    }

    return H_total, variables


def decode_solution(sample: dict, variables: dict):
    """
    Çözücüden gelen binary sample'ı (dict: {değişken_adı: 0/1}) fiziksel
    amplitude ve phase dizilerine çevirir.
    
    sample: SA/QAOA çözücüsünden gelen {değişken_label: 0 veya 1} sözlüğü
    variables: build_qubo_onehot()'tan dönen variables dict'i
    
    Dönüş: (amplitudes: np.array, phases_deg: np.array, n_violations: int)
    """
    N = variables["N"]
    n_phase = variables["n_phase"]
    n_amp = variables["n_amp"]
    phase_levels = variables["phase_levels"]
    amp_levels = variables["amp_levels"]

    amplitudes = np.zeros(N)
    phases_deg = np.zeros(N)
    n_violations = 0

    for n in range(N):
        # Faz: one-hot'ta hangi bit 1?
        phase_bits = [sample.get(f"x_phase[{n}][{k}]", 0) for k in range(n_phase)]
        active_phase = [k for k, v in enumerate(phase_bits) if v == 1]
        if len(active_phase) == 1:
            phases_deg[n] = phase_levels[active_phase[0]]
        else:
            # Kısıt ihlali: 0 veya >1 bit aktif → en yüksek olasılıklıyı/ilkini al
            n_violations += 1
            phases_deg[n] = phase_levels[active_phase[0]] if active_phase else 0.0

        # Genlik: aynı mantık
        amp_bits = [sample.get(f"x_amp[{n}][{k}]", 0) for k in range(n_amp)]
        active_amp = [k for k, v in enumerate(amp_bits) if v == 1]
        if len(active_amp) == 1:
            amplitudes[n] = amp_levels[active_amp[0]]
        else:
            n_violations += 1
            amplitudes[n] = amp_levels[active_amp[0]] if active_amp else 1.0

    return amplitudes, phases_deg, n_violations
