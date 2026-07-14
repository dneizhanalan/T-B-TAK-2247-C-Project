# Bu dosyada sistemin denge noktasini (r,v,Ca,h,IP3 hic degismedigi durum)
# ve bu noktanin kararli olup olmadigini buluyorum.

# scipy'nin fsolve fonksiyonu denklemleri sifira esitleyen noktayi ariyor.
# Ben once rastgele bir baslangic degeriyle denedim ama fiziksel olmayan
# (negatif kalsiyum gibi) bir sonuc verdi. Biraz deneme yanilma yaptiktan
# sonra asagidaki baslangic degeriyle duzgun bir sonuc buldum.

import numpy as np
from scipy.optimize import fsolve
from model import get_params, turevler


def denklemler(x, p):
    # tau_d=0 icin r_gecikmeli = r
    return turevler(x, x[0], p)


def denge_bul(p, tahmin=None):
    if tahmin is None:
        # deneme yanilmayla bulunmus iyi bir baslangic noktasi
        tahmin = np.array([0.8, -0.2, 0.68, 0.55, 2.9])

    sonuc = fsolve(denklemler, tahmin, args=(p,))
    return sonuc


def jacobian_hesapla(f, x, p, adim=1e-6):
    # sayisal turev ile jacobian matrisi (5x5) hesapliyorum
    n = len(x)
    J = np.zeros((n, n))
    f0 = f(x, p)
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = adim
        f1 = f(x + dx, p)
        J[:, j] = (f1 - f0) / adim
    return J


if __name__ == "__main__":
    p = get_params()
    x_star = denge_bul(p)

    isimler = ["r", "v", "Ca", "h", "IP3"]
    print("Denge noktasi:")
    for isim, deger in zip(isimler, x_star):
        print(f"  {isim} = {deger:.5f}")

    kalan = denklemler(x_star, p)
    print("kalan hata (0'a yakin olmali):", np.linalg.norm(kalan))

    J = jacobian_hesapla(denklemler, x_star, p)
    ozdegerler = np.linalg.eigvals(J)

    print("\nOzdegerler:")
    for oz in ozdegerler:
        print(" ", oz)

    max_reel = max(oz.real for oz in ozdegerler)
    if max_reel < 0:
        print(f"\nEn buyuk reel kisim = {max_reel:.6f}  -> sistem KARARLI")
    else:
        print(f"\nEn buyuk reel kisim = {max_reel:.6f}  -> sistem KARARSIZ")
