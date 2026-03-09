import numpy as np
import pandas as pd
from scipy.stats import qmc

# --- AYARLAR ---
num_samples = 300
num_criteria = 6
top_n = 20

# LHS örnekleme
sampler = qmc.LatinHypercube(d=num_criteria)
sample = sampler.random(n=num_samples)

# Parametre aralıkları
l_bounds = [6.0, 10.0, 0.0, 0.0, 0.0, 0.0]
u_bounds = [10.0, 20.0, 1.0, 1.0, 1.0, 1.0]

parameter_sets = qmc.scale(sample, l_bounds, u_bounds)


def validate_and_calculate(x):

    M, beta_deg, x1, x2, x3, x4 = x

    # --- SABİTLER ---
    gamma = 1.4
    R_gas = 287
    T_inf = 226.5
    rho_inf = 0.3639
    R_n = 0.05
    k = 1.83e-4

    beta = np.radians(beta_deg)

    if (x1 + x2) > 1.8:
        return None

    beta_min = np.arcsin(1/M)

    if beta <= beta_min:
        return None

    try:

        # --- HIZ ---
        a_inf = np.sqrt(gamma * R_gas * T_inf)
        V_inf = M * a_inf

        # --- ISI AKISI (Sutton-Graves) ---
        q_max = k * np.sqrt(rho_inf / R_n) * V_inf**3

        # --- DINAMIK BASINC ---
        q_inf = 0.5 * rho_inf * V_inf**2

        # --- PANEL MODEL ---
        V_hat = np.array([1,0,0])

        panel_normals = [
            np.array([0,0,1]),
            np.array([0.2,0,0.98]),
            np.array([-0.2,0,0.98]),
            np.array([0,0.2,0.98])
        ]

        panel_normals = [n/np.linalg.norm(n) for n in panel_normals]

        A_panels = [
            1 + x1,
            1 + x2,
            1 + x3,
            1 + x4
        ]

        F_total = np.zeros(3)

        for n_hat, A_i in zip(panel_normals, A_panels):

            theta = np.arccos(np.clip(np.dot(V_hat, n_hat), -1, 1))

            Cp = 2 * np.sin(theta)**2

            Fi = -Cp * q_inf * A_i * n_hat

            F_total += Fi

        # --- LIFT / DRAG YÖNLERI ---
        e_L = np.array([0,0,1])
        e_D = np.array([-1,0,0])

        Lift = np.dot(F_total, e_L)
        Drag = np.dot(F_total, e_D)

        if Drag <= 0:
            return None

        # --- REFERANS ALAN ---
        S_ref = sum(A_panels)

        CL = Lift / (q_inf * S_ref)
        CD = Drag / (q_inf * S_ref)

        LD = CL / CD

        # --- HACIM METRIGI ---
        length = 5
        span = x1 * length
        height = x2 * length

        volume = 0.5 * span * height * length

        return [LD, q_max, volume, V_inf, CL, CD]

    except:
        return None


results = []

for p in parameter_sets:

    outputs = validate_and_calculate(p)

    if outputs:
        results.append(list(p) + outputs)


columns = [
    'Mach','Beta','X1','X2','X3','X4',
    'L/D',
    'Max_Isi_Akisi',
    'Hacim',
    'V_inf',
    'CL',
    'CD'
]

df = pd.DataFrame(results, columns=columns)


# --- EN IYI TASARIMLAR ---
df_sorted = df.sort_values(by='L/D', ascending=False).reset_index(drop=True)

top_20_df = df_sorted.head(top_n)


df_sorted.to_csv("waverider_gercekci_analiz.csv", index=False)
top_20_df.to_csv("waverider_en_iyi_20_aday.csv", index=False)

print(f"Geçerli tasarım sayısı: {len(df)}")
print(f"En iyi {top_n} tasarım kaydedildi.")
