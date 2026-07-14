
# Butun dosyalari sirayla calistirip grafikleri figs/ klasorune kaydediyor.


import subprocess
import sys

dosyalar = [
    "denge_noktasi.py",
    "dt_yakinsama_testi.py",
    "spektral_analiz.py",
    "gecikme_taramasi.py",
    "rejim_haritasi.py",
]

for dosya in dosyalar:
    print("\n" + "=" * 60)
    print("calisiyor:", dosya)
    print("=" * 60)
    sonuc = subprocess.run([sys.executable, dosya])
    if sonuc.returncode != 0:
        print("UYARI:", dosya, "hata verdi")

print("\nhepsi bitti, grafikler figs/ klasorunde")
