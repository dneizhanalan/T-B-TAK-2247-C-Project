# model.py
# Burada QIF noron populasyonu (MPR ortalama-alan modeli) ile
# astrosit kalsiyum modelini (Li-Rinzel) birlestirdim.
#
# Notron kismi: r = ates etme orani, v = ortalama membran potansiyeli
# Astrosit kismi: Ca = kalsiyum, h = IP3 receptor degiskeni, IP3 = IP3 miktari
#
# Notron ile astrosit birbirini etkiliyor:
#   - notron aktif oldukca (r buyudukce) astrositte IP3 uretimi artiyor
#   - astrositteki kalsiyum arttikca notronlar arasi baglanti gucu (J) artiyor
#
# NOT: zaman birimini hep milisaniye (ms) olarak kullandim. Li-Rinzel
# modelinin orjinal parametreleri saniye cinsindendi, o yuzden asagida
# 1/1000 ile carpip ms'ye cevirdim (yoksa astrosit cok yavas/hizli
# calisiyordu, hoca da dedi ki bu tarz birim hatalarina dikkat et diye).

import numpy as np

# --- parametreler (hepsi tek yerde, degistirmek kolay olsun diye) ---

def get_params():
    p = {}

    p["tau_m"] = 10.0     # noron zaman sabiti (ms), gama bant icin lazim
    p["Delta"] = 1.0       # noron populasyonundaki cesitlilik
    p["eta"]   = -1.25     # noronlarin ortalama uyarilma seviyesi
    p["J0"]    = 8.0        # temel baglanti gucu
    p["k_M"]   = 2.0        # astrositin noronlara etkisinin gucu
    p["tau_d"] = 0.0        # sinaptik gecikme (ms) - bunu degistirerek deney yapiyoruz

    # Li-Rinzel astrosit modeli parametreleri (kitaptaki/makaledeki
    # standart degerler, biz degistirmedik)
    p["v1"] = 6.0
    p["v2"] = 0.11
    p["v3"] = 0.9
    p["k3"] = 0.1
    p["d1"] = 0.13
    p["d2"] = 1.049
    p["d3"] = 0.9434
    p["d5"] = 0.08234
    p["a2"] = 0.2
    p["c0"] = 2.0
    p["c1"] = 0.185

    p["IP3_0"] = 0.16          # dinlenme halindeki IP3
    p["tau_ip3"] = 7000.0      # IP3'un yavas yavas eski haline donme suresi (ms)
    p["eps_glu"] = 0.0005      # noron aktivitesinin IP3'e ne kadar etki ettigi

    return p


def baslangic_durumu(p):
    # simulasyona baslarken kullandigimiz ilk degerler
    # (cok da onemli degil, zaman icinde denge noktasina yaklasiyor)
    r0 = 0.05
    v0 = -1.0
    Ca0 = 0.1
    h0 = 0.8
    IP3_0 = p["IP3_0"]
    return np.array([r0, v0, Ca0, h0, IP3_0])


def turevler(x, r_gecikmeli, p):
    # x = [r, v, Ca, h, IP3]
    # r_gecikmeli: gecikme varsa r(t - tau_d), yoksa direkt r
    r, v, Ca, h, IP3 = x

    J_etkin = p["J0"] + p["k_M"] * Ca   # astrosit noron baglantisini guclendiriyor

    dr = (p["Delta"] / np.pi + 2.0 * r * v) / p["tau_m"]
    dv = (v**2 - (np.pi * r)**2 + p["eta"] + J_etkin * r_gecikmeli) / p["tau_m"]

    # --- Li-Rinzel astrosit denklemleri ---
    m_inf = (IP3 / (IP3 + p["d1"])) * (Ca / (Ca + p["d5"]))
    Q2 = p["d2"] * (IP3 + p["d1"]) / (IP3 + p["d3"])
    h_inf = Q2 / (Q2 + Ca)
    tau_h = 1.0 / (p["a2"] * (Q2 + Ca))   # bu saniye cinsinden

    Ca_er = (p["c0"] - Ca) / p["c1"]
    J_kanal = p["c1"] * p["v1"] * (m_inf**3) * (h**3) * (Ca_er - Ca)
    J_pompa = p["v3"] * (Ca**2) / (Ca**2 + p["k3"]**2)
    J_sizinti = p["c1"] * p["v2"] * (Ca_er - Ca)

    dCa_saniye = J_kanal + J_sizinti - J_pompa
    dh_saniye = (h_inf - h) / tau_h

    # saniyeyi ms'ye ceviriyoruz (1 s = 1000 ms)
    dCa = dCa_saniye / 1000.0
    dh = dh_saniye / 1000.0

    dIP3 = (p["IP3_0"] - IP3) / p["tau_ip3"] + p["eps_glu"] * r

    return np.array([dr, dv, dCa, dh, dIP3])
