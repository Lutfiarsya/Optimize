import numpy as np
from scipy.optimize import minimize

def profit(x):
    x1, x2 = x
    return 6*x1 - 0.1*x1**2 + 8*x2 - 0.2*x2**2


hasil7 = minimize(
    lambda x: -profit(x),
    x0=[1, 1],                                   
    constraints={"type": "eq",
                 "fun": lambda x: x[0] + 2*x[1] - 40},
    bounds=[(0, None), (0, None)],               
    method="SLSQP",
)

x1, x2 = hasil7.x
print("Soal no 7")
print(f"x1 (Kursi) = {x1:.0f}")
print(f"x2 (Meja)  = {x2:.0f}")
print(f"Z (Profit) = {profit(hasil7.x):.0f} Juta Rupiah")
print(f"Jam mesin  = {x1 + 2*x2:.0f} Jam")


# Koordinat berurutan A, B, C, D
toko = np.array([[2, 8], [6, 2], [8, 6], [3, 3]], dtype=float) 
# Pengiriman
w = np.array([20, 10, 30, 15], dtype=float)

def total_jarak(p):
    x, y = p
    return np.sum(w * np.sqrt((x - toko[:, 0])**2 + (y - toko[:, 1])**2))

# Formulasi Bagian A
pusat = (w[:, None] * toko).sum(axis=0) / w.sum()
hasil8 = minimize(total_jarak, x0=pusat, method="Nelder-Mead",
                  options={"xatol": 1e-10, "fatol": 1e-10})

# Output Bagian A
print("\nSoal no 8 Bagian A")
print(f"Lokasi gudang optimal = ({hasil8.x[0]:.4f}, {hasil8.x[1]:.4f})")
print(f"Total jarak tertimbang = {hasil8.fun:.4f}")


# Output Bagian B
# Titik Pusat Gravitasi
print("\nSoal no 8 bagian B")
print(f"Pusat gravitasi = ({pusat[0]:.4f}, {pusat[1]:.4f})")
print(f"Total jarak tertimbang = {total_jarak(pusat):.4f}")
print(f"Selisih = {total_jarak(pusat) - hasil8.fun:.4f}")