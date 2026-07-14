# rk4_cozucu.py
# Basit RK4 (Runge-Kutta 4. derece) entegratoru.
# Hocamin dedigine gore RK4 Euler'den cok daha dogru sonuc veriyor,
# o yuzden onu kullandim.
#
# Gecikme (tau_d) varsa, r'nin gecmisteki degerini bir dizide (X)
# tutup oradan okuyorum. Bu yonteme "method of steps" deniyormus,
# yani gecikmeyi adim adim gecmisten okuyarak halletmis oluyoruz.

import numpy as np
from model import turevler


def simule_et(p, x0, t_max, dt):
    n_adim = int(t_max / dt)
    t = np.zeros(n_adim + 1)
    X = np.zeros((n_adim + 1, len(x0)))
    X[0] = x0

    tau_d = p["tau_d"]
    gecikme_adim = int(round(tau_d / dt)) if tau_d > 0 else 0

    for i in range(n_adim):
        xi = X[i]
        t[i + 1] = t[i] + dt

        if gecikme_adim == 0:
            # gecikme yok, r'nin kendisini kullaniyoruz
            k1 = turevler(xi, xi[0], p)
            x2 = xi + 0.5 * dt * k1
            k2 = turevler(x2, x2[0], p)
            x3 = xi + 0.5 * dt * k2
            k3 = turevler(x3, x3[0], p)
            x4 = xi + dt * k3
            k4 = turevler(x4, x4[0], p)
        else:
            # gecikmeli r degerini gecmis dizisinden aliyoruz
            j = i - gecikme_adim
            if j < 0:
                r_gecmis = x0[0]
            else:
                r_gecmis = X[j, 0]

            k1 = turevler(xi, r_gecmis, p)
            k2 = turevler(xi + 0.5 * dt * k1, r_gecmis, p)
            k3 = turevler(xi + 0.5 * dt * k2, r_gecmis, p)
            k4 = turevler(xi + dt * k3, r_gecmis, p)

        X[i + 1] = xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, X
