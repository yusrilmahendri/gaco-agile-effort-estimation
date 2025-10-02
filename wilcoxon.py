import numpy as np
from scipy.stats import wilcoxon, rankdata
import pandas as pd

# Estimasi effort hasil Algoritma Genetika (Algen)
estEffort_algen = [
    4922.80909, 635.056357, 1595.122799, 12730.48117, 2332.396446,
    1902.899019, 5467.739143, 6120.645915, 9515.857127, 3031.758876,
    2925.696729, 715.39284, 484.6342048, 1965.276065, 2944.782741,
    1217.056816, 13978.09507, 26695.70585, 10648.69769, 7407.579548,
    10324.13994, 6682.466398, 4894.217458, 5344.03798, 3605.692305,
    28805.09419, 1147.357289, 1096.931173, 1293.340491, 1192.154262,
    2221.850411, 768.9755224, 892.2902272, 2365.056975, 6083.031,
    4993.476326, 3036.064659, 11421.75561, 4993.620687, 7977.105098,
    992.5988843, 4870.980658, 951.3883239, 4769.551748, 6682.200693,
    8028.718375, 707.9818663, 9121.34065, 1552.765328, 4851.029963,
    3953.739372, 3162.338502, 3935.016426, 7698.497188, 3070.597794,
    1701.957774, 10806.23678, 6261.490982, 5859.163415, 4059.836332,
    6501.923677, 29602.09309
]

# Estimasi effort hasil Genetic Algorithm + ACO (GACO)
estEffort_gaco = [
    647.072031, 129.9276801, 253.9839432, 1055.882666, 382.9680742,
    345.0114111, 822.4441316, 1240.844265, 1657.74296, 484.3363303,
    739.0635308, 107.9687193, 128.7990153, 358.8136597, 370.9928591,
    210.9808581, 1849.440761, 2474.82118, 961.8781399, 671.1280756,
    2953.282589, 501.382805, 915.5972138, 873.7065361, 463.3890854,
    3920.446804, 253.0138188, 195.5937809, 248.2413269, 386.9880913,
    424.826989, 204.0222912, 177.2536168, 839.9512285, 1647.837104,
    1035.147986, 547.8520413, 2734.364649, 1459.571296, 2546.128826,
    434.2032316, 2084.939997, 529.5821319, 2315.69983, 2848.310336,
    3214.166675, 318.1100312, 2963.140759, 504.6854668, 1570.495442,
    1645.947751, 1046.465096, 2228.492384, 2598.512082, 1038.510068,
    1003.335943, 2699.877999, 2255.310045, 1946.446009, 1881.109259,
    2210.261651, 9409.074245
]

# --- 1. Hitung Perbedaan ---
# Perbedaan: estEffort_algen - estEffort_gaco
diff = np.array(estEffort_algen) - np.array(estEffort_gaco)
N_total = len(diff)

# --- 2. Filter Perbedaan Nol (Ties) ---
non_zero_diff = diff[diff != 0]
N_non_zero = len(non_zero_diff)
N_ties = N_total - N_non_zero

# --- 3. Hitung Rank dari Nilai Absolut ---
abs_diff = np.abs(non_zero_diff)
# Gunakan rankdata untuk peringkat dengan ties
ranks = rankdata(abs_diff)

# --- 4. Pisahkan Ranks berdasarkan Tanda ---
# Ranks Positif (algen > gaco, selisih > 0)
positive_ranks = ranks[non_zero_diff > 0]
N_pos = len(positive_ranks)
sum_pos_ranks = np.sum(positive_ranks)
mean_pos_rank = sum_pos_ranks / N_pos if N_pos > 0 else 0

# Ranks Negatif (algen < gaco, selisih < 0)
negative_ranks = ranks[non_zero_diff < 0]
N_neg = len(negative_ranks)
sum_neg_ranks = np.sum(negative_ranks)
mean_neg_rank = sum_neg_ranks / N_neg if N_neg > 0 else 0


# --- 5. Format Output Tabel ---
data = {
    'N': [N_neg, N_pos, N_ties, N_total],
    'Mean Rank': [f"{mean_neg_rank:.2f}" if N_neg > 0 else ".00", 
                  f"{mean_pos_rank:.2f}" if N_pos > 0 else ".00", 
                  ".00", 
                  ""],
    'Sum of Ranks': [f"{sum_neg_ranks:.2f}", f"{sum_pos_ranks:.2f}", ".00", ""],
}

index_labels = ['Negative Ranks', 'Positive Ranks', 'Ties', 'Total']
df_ranks = pd.DataFrame(data, index=index_labels)

# --- 6. Tampilkan Hasil Uji Signifikansi ---
stat, p = wilcoxon(estEffort_algen, estEffort_gaco)
alpha = 0.05
result_interpretation = "➡️ Ada perbedaan signifikan estimasi effort (H0 ditolak)" if p < alpha else "➡️ Tidak ada perbedaan signifikan estimasi effort (H0 diterima)"

print("Tabel Ranks Wilcoxon (estEffort_algen - estEffort_gaco)\n")
# Mengatur format tampilan pandas agar mirip tabel
pd.set_option('display.colheader_justify', 'center')
print(df_ranks)
print("\n" + "="*50)
print(f"Statistik Wilcoxon (T): {stat:.2f}")
print(f"p-value: {p:.10f}")
print(result_interpretation)