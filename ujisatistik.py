from scipy.stats import wilcoxon
import pandas as pd
import statistics


# ============================================================
# Data Absolute Error Dataset Maxwell
# ============================================================
ae_algen_maxwell = [
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

ae_gaco_maxwell = [
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


#
# Catatan:
# Data Ziauddin di bawah ini belum digunakan dalam output karena masih bersifat sementara.
# Setelah data AE GA dan GACO Ziauddin sudah final, aktifkan kembali bagian Tabel 4.3.
# ============================================================
# Data Absolute Error Dataset Ziauddin
# ============================================================
ae_algen_zia = [
    58.00350052, 80.97989152, 51.94129062, 45.84534862,
    34.60792893, 106.5267196, 28.88216303, 84.70855991,
    35.00129577, 66.02780003, 40.99201217, 38.96668158,
    35.00644373, 25.98593576, 21.99470607, 102.9807793,
    40.01291212, 49.98487598, 76.02435325, 50.99512232,
    33.97562046
]

ae_gaco_zia = [
    62.99117734, 92.01368609, 55.99461247, 63.76621792,
    31.99396456, 83.0151176, 35.00755978, 92.99996588,
    35.99916051, 62.01644315, 45.00041312, 37.01291564,
    31.9964652, 30.00113685, 20.99666635, 111.9804098,
    38.99929481, 51.9868647, 79.99974037, 56.01685318,
    34.99340775
]


# ============================================================
# Fungsi bantu uji Wilcoxon dan ringkasan statistik
# ============================================================
def validate_data(dataset_name, ae_ga, ae_gaco):
    if len(ae_ga) != len(ae_gaco):
        raise ValueError(
            f"Jumlah data pada dataset {dataset_name} tidak sama: "
            f"GA = {len(ae_ga)}, GACO = {len(ae_gaco)}"
        )

    if len(ae_ga) == 0:
        raise ValueError(f"Data AE pada dataset {dataset_name} tidak boleh kosong.")


def run_wilcoxon(dataset_name, ae_ga, ae_gaco, table_number):
    validate_data(dataset_name, ae_ga, ae_gaco)

    mae_ga = sum(ae_ga) / len(ae_ga)
    mae_gaco = sum(ae_gaco) / len(ae_gaco)

    median_ga = statistics.median(ae_ga)
    median_gaco = statistics.median(ae_gaco)

    std_ga = statistics.stdev(ae_ga) if len(ae_ga) > 1 else 0.0
    std_gaco = statistics.stdev(ae_gaco) if len(ae_gaco) > 1 else 0.0

    differences = [ga - gaco for ga, gaco in zip(ae_ga, ae_gaco)]

    jumlah_gaco_lebih_baik = sum(1 for diff in differences if diff > 0)
    jumlah_ga_lebih_baik = sum(1 for diff in differences if diff < 0)
    jumlah_sama = sum(1 for diff in differences if diff == 0)

    # Hipotesis penelitian:
    # H0: Tidak terdapat perbedaan signifikan antara hasil estimasi effort GA dan GACO.
    # H1: Terdapat perbedaan signifikan antara hasil estimasi effort GA dan GACO.
    # Karena H1 tidak menyatakan arah tertentu, maka digunakan Wilcoxon two-sided.
    stat_two_sided, p_two_sided = wilcoxon(
        ae_ga,
        ae_gaco,
        alternative="two-sided",
        zero_method="wilcox"
    )

    alpha = 0.05
    keputusan = "H0 ditolak" if p_two_sided < alpha else "H0 gagal ditolak"
    kesimpulan = (
        "Terdapat perbedaan signifikan"
        if p_two_sided < alpha
        else "Tidak terdapat perbedaan signifikan"
    )

    arah_perbandingan = (
        "GACO memiliki MAE lebih rendah dibandingkan GA"
        if mae_gaco < mae_ga
        else "GA memiliki MAE lebih rendah dibandingkan GACO"
        if mae_ga < mae_gaco
        else "GA dan GACO memiliki MAE yang sama"
    )

    table_rows = [
        {
            "Dataset": dataset_name,
            "Metode": "Algoritma Genetika (GA)",
            "N": len(ae_ga),
            "Rata-rata AE / MAE": round(mae_ga, 3),
            "Median AE": round(median_ga, 3),
            "Standar Deviasi AE": round(std_ga, 3),
            "Statistik Wilcoxon": round(stat_two_sided, 3),
            "p-value": p_two_sided,
            "Keputusan": keputusan,
        },
        {
            "Dataset": dataset_name,
            "Metode": "Genetic Algorithm + ACO (GACO)",
            "N": len(ae_gaco),
            "Rata-rata AE / MAE": round(mae_gaco, 3),
            "Median AE": round(median_gaco, 3),
            "Standar Deviasi AE": round(std_gaco, 3),
            "Statistik Wilcoxon": "",
            "p-value": "",
            "Keputusan": kesimpulan,
        },
    ]

    comparison_row = {
        "Tabel": table_number,
        "Dataset": dataset_name,
        "Jumlah Data": len(ae_ga),
        "MAE GA": mae_ga,
        "MAE GACO": mae_gaco,
        "Selisih MAE (GA - GACO)": mae_ga - mae_gaco,
        "Persentase Penurunan MAE (%)": ((mae_ga - mae_gaco) / mae_ga) * 100 if mae_ga != 0 else 0.0,
        "Median AE GA": median_ga,
        "Median AE GACO": median_gaco,
        "Standar Deviasi AE GA": std_ga,
        "Standar Deviasi AE GACO": std_gaco,
        "Jumlah Proyek GACO Lebih Baik": jumlah_gaco_lebih_baik,
        "Jumlah Proyek GA Lebih Baik": jumlah_ga_lebih_baik,
        "Jumlah Proyek Sama": jumlah_sama,
        "Statistik Wilcoxon": stat_two_sided,
        "p-value": p_two_sided,
        "Keputusan": keputusan,
        "Kesimpulan": kesimpulan,
        "Arah Perbandingan MAE": arah_perbandingan,
    }

    detail_rows = []
    for index, (ga, gaco, diff) in enumerate(zip(ae_ga, ae_gaco, differences), start=1):
        detail_rows.append({
            "Dataset": dataset_name,
            "Proyek": index,
            "AE GA": ga,
            "AE GACO": gaco,
            "Selisih AE (GA - GACO)": diff,
            "Metode Lebih Baik": "GACO" if diff > 0 else "GA" if diff < 0 else "Sama"
        })

    return table_rows, comparison_row, detail_rows


#
# Untuk saat ini hanya dataset Maxwell yang digunakan karena data Ziauddin belum final.
# Jika data Ziauddin sudah final, tambahkan kembali baris:
# ("Ziauddin", ae_algen_zia, ae_gaco_zia, "Tabel 4.3"),
datasets = [
    ("Maxwell", ae_algen_maxwell, ae_gaco_maxwell, "Tabel 4.2"),
]

table_rows = []
comparison_rows = []
detail_rows = []

for dataset_name, ae_ga, ae_gaco, table_number in datasets:
    table_result, comparison, detail = run_wilcoxon(dataset_name, ae_ga, ae_gaco, table_number)
    table_rows.extend(table_result)
    comparison_rows.append(comparison)
    detail_rows.extend(detail)


df_tables = pd.DataFrame(table_rows)
df_comparison = pd.DataFrame(comparison_rows)
df_detail = pd.DataFrame(detail_rows)


# ============================================================
# Export hasil ke CSV dan Excel
# ============================================================
df_tables.to_csv("tabel_4_2_wilcoxon_maxwell.csv", index=False)
df_comparison.to_csv("ringkasan_wilcoxon_maxwell.csv", index=False)
df_detail.to_csv("detail_ae_maxwell.csv", index=False)

with pd.ExcelWriter("tabel_4_2_wilcoxon_maxwell.xlsx") as writer:
    df_tables.to_excel(writer, sheet_name="Tabel 4.2", index=False)
    df_comparison.to_excel(writer, sheet_name="Ringkasan", index=False)
    df_detail.to_excel(writer, sheet_name="Detail AE", index=False)


# ============================================================
# Print hasil ke terminal
# ============================================================
print("===== TABEL 4.2 HASIL UJI WILCOXON SIGNED-RANK TEST DATASET MAXWELL =====")
print(df_tables.to_string(index=False))
print()
print("===== RINGKASAN INTERPRETASI =====")
row = df_comparison.iloc[0]
print(f"Dataset                         : {row['Dataset']}")
print(f"Jumlah data                     : {row['Jumlah Data']}")
print(f"MAE GA                          : {row['MAE GA']:.3f}")
print(f"MAE GACO                        : {row['MAE GACO']:.3f}")
print(f"Selisih MAE (GA - GACO)         : {row['Selisih MAE (GA - GACO)']:.3f}")
print(f"Persentase penurunan MAE        : {row['Persentase Penurunan MAE (%)']:.3f}%")
print(f"Jumlah proyek GACO lebih baik   : {row['Jumlah Proyek GACO Lebih Baik']}")
print(f"Jumlah proyek GA lebih baik     : {row['Jumlah Proyek GA Lebih Baik']}")
print(f"Statistik Wilcoxon              : {row['Statistik Wilcoxon']:.3f}")
print(f"p-value                         : {row['p-value']}")
print(f"Keputusan                       : {row['Keputusan']}")
print(f"Kesimpulan                      : {row['Kesimpulan']}")
print()
print("File berhasil disimpan:")
print("- tabel_4_2_wilcoxon_maxwell.csv")
print("- ringkasan_wilcoxon_maxwell.csv")
print("- detail_ae_maxwell.csv")
print("- tabel_4_2_wilcoxon_maxwell.xlsx")
