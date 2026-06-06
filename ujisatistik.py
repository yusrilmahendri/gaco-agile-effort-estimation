import csv
import math
import statistics


# ============================================================
# Fungsi statistik tanpa scipy
# ============================================================
def rankdata(values):
    """
    Menghitung ranking dengan metode average rank untuk nilai yang sama.
    Fungsi ini digunakan sebagai pengganti scipy.stats.rankdata.
    """
    indexed_values = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0

    while i < len(indexed_values):
        j = i
        while j + 1 < len(indexed_values) and indexed_values[j + 1][0] == indexed_values[i][0]:
            j += 1

        average_rank = (i + 1 + j + 1) / 2.0

        for k in range(i, j + 1):
            original_index = indexed_values[k][1]
            ranks[original_index] = average_rank

        i = j + 1

    return ranks


def normal_cdf(z):
    """
    Menghitung cumulative distribution function untuk distribusi normal standar.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon(ae_ga, ae_gaco, alternative="two-sided", zero_method="wilcox"):
    """
    Menghitung Wilcoxon Signed-Rank Test tanpa scipy.

    Parameter alternative dan zero_method dipertahankan agar pemanggilan fungsi
    tetap sama seperti scipy.stats.wilcoxon pada kode sebelumnya.
    """
    if alternative != "two-sided":
        raise ValueError("Fungsi ini hanya mendukung alternative='two-sided'.")

    if zero_method != "wilcox":
        raise ValueError("Fungsi ini hanya mendukung zero_method='wilcox'.")

    differences = [ga - gaco for ga, gaco in zip(ae_ga, ae_gaco)]
    non_zero_differences = [diff for diff in differences if diff != 0]

    if len(non_zero_differences) == 0:
        return 0.0, 1.0

    abs_differences = [abs(diff) for diff in non_zero_differences]
    ranks = rankdata(abs_differences)

    positive_rank_sum = sum(
        rank for diff, rank in zip(non_zero_differences, ranks) if diff > 0
    )
    negative_rank_sum = sum(
        rank for diff, rank in zip(non_zero_differences, ranks) if diff < 0
    )

    statistic = min(positive_rank_sum, negative_rank_sum)
    n = len(non_zero_differences)

    # Exact p-value digunakan untuk data kecil tanpa tied ranks.
    ranks_are_integer = all(abs(rank - round(rank)) < 1e-12 for rank in ranks)

    if n <= 25 and ranks_are_integer:
        integer_ranks = [int(round(rank)) for rank in ranks]
        total_rank_sum = sum(integer_ranks)
        target_statistic = int(round(statistic))

        counts = {0: 1}
        for rank in integer_ranks:
            next_counts = counts.copy()
            for current_sum, count in counts.items():
                new_sum = current_sum + rank
                next_counts[new_sum] = next_counts.get(new_sum, 0) + count
            counts = next_counts

        extreme_count = 0
        total_combinations = 2 ** n
        for rank_sum, count in counts.items():
            if min(rank_sum, total_rank_sum - rank_sum) <= target_statistic:
                extreme_count += count

        p_value = min(1.0, extreme_count / total_combinations)
        return float(statistic), p_value

    # Normal approximation digunakan untuk data besar.
    mean_rank_sum = n * (n + 1) / 4.0
    variance_rank_sum = n * (n + 1) * (2 * n + 1) / 24.0
    standard_deviation = math.sqrt(variance_rank_sum)

    if statistic < mean_rank_sum:
        z_score = (statistic - mean_rank_sum + 0.5) / standard_deviation
    else:
        z_score = (statistic - mean_rank_sum - 0.5) / standard_deviation

    p_value = 2.0 * min(normal_cdf(z_score), 1.0 - normal_cdf(z_score))
    return float(statistic), p_value


# ============================================================
# Data Absolute Error Dataset Maxwell
# ============================================================
ae_algen_maxwell = [
    1453.6146631430515,
    68.10361763959511,
    520.0176136351073,
    4035.3452885180322,
    389.60959455313446,
    601.1659140123028,
    1207.9935853300535,
    2349.921732923688,
    2926.3136736392844,
    945.5006695453865,
    522.0224321079725,
    358.9649474393213,
    248.91679685133965,
    660.5158470607474,
    823.8689118472698,
    277.7056413497287,
    7507.405162274341,
    4876.472150366699,
    3444.18856467141,
    1892.3838744955942,
    2026.381887487346,
    2015.1301826562776,
    2421.8068108990606,
    3206.7758734435433,
    1958.3049467343753,
    5998.804854457932,
    212.07327111490423,
    365.10806913170507,
    703.1969689756728,
    361.70937218909785,
    526.7178840234106,
    116.88529690637745,
    191.9807306481096,
    0.05497518026754733,
    257.39168668411116,
    668.6125248366882,
    707.3979566715179,
    4562.075785372979,
    3134.9462560867523,
    4348.195144547746,
    533.5885205996326,
    3183.4759872552454,
    580.365754538055,
    3429.1790383074604,
    5049.143318730262,
    3854.769682732167,
    443.7090464025496,
    6234.16211086505,
    972.9960759355574,
    2117.0050823801835,
    1851.2102067700375,
    1420.3579884129604,
    2185.4309230657573,
    4433.099447469027,
    1842.240858308718,
    818.8258704924913,
    6304.785768821387,
    3472.768693299179,
    2682.326736439091,
    2858.785397682399,
    3947.157150647504,
    16710.343358911792,
]

ae_gaco_maxwell = [
    0.43159543970341474,
    0.028653023685592416,
    0.027482880725699488,
    202.94555614397223,
    289.3190161275695,
    0.23415631683241145,
    474.8338578034037,
    453.5689389736244,
    687.7522456707027,
    132.25407350517452,
    0.24273185385902707,
    1.0749530972477288,
    70.19526967340622,
    8.185687649970532,
    95.2503002655165,
    58.94646618380705,
    1.0721458526375045,
    1.0664010655382299,
    463.03526997049266,
    1095.1041088026914,
    0.4857140556587183,
    130.4749191018198,
    306.4511327024402,
    238.77379675360748,
    0.08910178680929448,
    0.24512229479023517,
    0.04470054032131543,
    0.01139538445056587,
    24.58416195379209,
    0.01999667878180844,
    0.08168563625696379,
    0.03334203988265472,
    44.87047732286787,
    0.40358626344846016,
    0.02884673479502453,
    0.03765212252051242,
    0.4604173129090441,
    1911.3944893327957,
    1312.1805008732617,
    1592.9769737216693,
    270.8666263794628,
    1838.2996757667297,
    444.2595892103781,
    2269.2314129979654,
    2400.4109115646834,
    1621.1926434494412,
    225.68049246475118,
    3892.177730696698,
    364.5555391870712,
    286.04163119395344,
    1061.0399118076625,
    695.4708724297484,
    850.3264827407529,
    3013.016125775554,
    939.3261683679909,
    544.2546637926217,
    2486.990334410414,
    1351.9005202624412,
    1268.292182289487,
    1288.9469636223016,
    1677.9279767609492,
    6409.638822523815,
]


# ============================================================
# Data Absolute Error Dataset Ziauddin
# ============================================================
ae_algen_zia = [
    0.02509048211584286,
    0.0023156018600758443,
    0.0077600736054606045,
    27.010826909858466,
    0.023904830461148663,
    56.04833321322228,
    0.0004883574653078426,
    0.007326521676432662,
    0.027185997366359516,
    0.003816992462610358,
    0.017663285690858288,
    0.02637258794075592,
    0.0032967151809373263,
    0.00785677338292956,
    0.008273320853341204,
    0.016209084242021277,
    0.016398198156124977,
    0.025175745417911344,
    0.021567529237245253,
    0.016475463658132128,
    0.016732581727488594,
]

ae_gaco_zia = [
    0.0036663034305348674,
    0.009085118331327635,
    0.001041792351564652,
    23.955837325262934,
    0.007668137332451863,
    50.6829268292683,
    0.0010750965282682046,
    0.00011479576296835603,
    0.002385975944086738,
    0.0043233390584092035,
    0.004609526942061848,
    0.0024781000322562363,
    0.0013733847097938678,
    0.0018060249931224348,
    0.0003522948156557959,
    0.0014294328793909017,
    0.0005184500227031208,
    0.004122102363709246,
    0.0068621292286081825,
    0.0011985255624082924,
    0.00022168203274475218,
]


# ============================================================
# Fungsi Wilcoxon Ranks Table
# ============================================================
def calculate_wilcoxon_ranks(dataset_name, ae_ga, ae_gaco, variable_label):
    """
    Menghitung tabel Ranks untuk Wilcoxon Signed-Rank Test.

    Selisih dihitung sebagai:
    AE GA - AE GACO

    Interpretasi:
    - Negative Ranks: AE GA < AE GACO, artinya GA lebih baik.
    - Positive Ranks: AE GA > AE GACO, artinya GACO lebih baik.
    - Ties: AE GA = AE GACO.
    """
    validate_data(dataset_name, ae_ga, ae_gaco)

    differences = [ga - gaco for ga, gaco in zip(ae_ga, ae_gaco)]
    non_zero_abs_diff = [abs(diff) for diff in differences if diff != 0]

    if len(non_zero_abs_diff) > 0:
        ranks = rankdata(non_zero_abs_diff)
    else:
        ranks = []

    rank_index = 0
    negative_ranks = []
    positive_ranks = []
    ties_count = 0

    for diff in differences:
        if diff < 0:
            negative_ranks.append(float(ranks[rank_index]))
            rank_index += 1
        elif diff > 0:
            positive_ranks.append(float(ranks[rank_index]))
            rank_index += 1
        else:
            ties_count += 1

    negative_n = len(negative_ranks)
    positive_n = len(positive_ranks)
    total_n = len(differences)

    negative_mean_rank = sum(negative_ranks) / negative_n if negative_n > 0 else 0.0
    positive_mean_rank = sum(positive_ranks) / positive_n if positive_n > 0 else 0.0

    negative_sum_rank = sum(negative_ranks)
    positive_sum_rank = sum(positive_ranks)

    rank_rows = [
        {
            "Dataset": dataset_name,
            "Variabel": variable_label,
            "Ranks": "Negative Ranks",
            "N": negative_n,
            "Mean Rank": round(negative_mean_rank, 2),
            "Sum of Ranks": round(negative_sum_rank, 2),
        },
        {
            "Dataset": dataset_name,
            "Variabel": variable_label,
            "Ranks": "Positive Ranks",
            "N": positive_n,
            "Mean Rank": round(positive_mean_rank, 2),
            "Sum of Ranks": round(positive_sum_rank, 2),
        },
        {
            "Dataset": dataset_name,
            "Variabel": variable_label,
            "Ranks": "Ties",
            "N": ties_count,
            "Mean Rank": 0.00,
            "Sum of Ranks": 0.00,
        },
        {
            "Dataset": dataset_name,
            "Variabel": variable_label,
            "Ranks": "Total",
            "N": total_n,
            "Mean Rank": "",
            "Sum of Ranks": "",
        },
    ]

    return rank_rows
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
            "Tabel": table_number,
            "Dataset": dataset_name,
            "Metode": "Algoritma Genetika (GA)",
            "N": len(ae_ga),
            "Rata-rata AE / MAE": round(mae_ga, 3),
            "Median AE": round(median_ga, 3),
            "Standar Deviasi AE": round(std_ga, 3),
            "Statistik Wilcoxon": round(stat_two_sided, 3),
            "Asymp. Sig. (2-tailed)": p_two_sided,
            "Keputusan": keputusan,
        },
        {
            "Tabel": table_number,
            "Dataset": dataset_name,
            "Metode": "Genetic Algorithm + ACO (GACO)",
            "N": len(ae_gaco),
            "Rata-rata AE / MAE": round(mae_gaco, 3),
            "Median AE": round(median_gaco, 3),
            "Standar Deviasi AE": round(std_gaco, 3),
            "Statistik Wilcoxon": "",
            "Asymp. Sig. (2-tailed)": "",
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
        "Asymp. Sig. (2-tailed)": p_two_sided,
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


# ============================================================
# Eksekusi uji Wilcoxon Signed-Rank Test untuk dua dataset
# ============================================================
datasets = [
    (
        "Maxwell",
        ae_algen_maxwell,
        ae_gaco_maxwell,
        "Tabel 4.3",
        "Estimasi usaha Maxwell GA - Estimasi usaha Maxwell GACO",
    ),
    (
        "Ziauddin",
        ae_algen_zia,
        ae_gaco_zia,
        "Tabel 4.4",
        "Estimasi usaha Agile GA - Estimasi usaha Agile GACO Ziauddin",
    ),
]

table_rows = []
rank_rows = []
comparison_rows = []
detail_rows = []

# ============================================================
# Export hasil ke CSV tanpa pandas/openpyxl
# ============================================================
def write_csv(filename, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows, columns):
    if not rows:
        return ""

    formatted_rows = []
    for row in rows:
        formatted_rows.append([str(row.get(column, "")) for column in columns])

    widths = []
    for index, column in enumerate(columns):
        max_cell_width = max(len(row[index]) for row in formatted_rows) if formatted_rows else 0
        widths.append(max(len(column), max_cell_width))

    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    separator = "-+-".join("-" * width for width in widths)
    body = "\n".join(
        " | ".join(row[index].ljust(widths[index]) for index in range(len(columns)))
        for row in formatted_rows
    )

    return header + "\n" + separator + "\n" + body


for dataset_name, ae_ga, ae_gaco, table_number, variable_label in datasets:
    table_result, comparison, detail = run_wilcoxon(dataset_name, ae_ga, ae_gaco, table_number)
    ranks_result = calculate_wilcoxon_ranks(dataset_name, ae_ga, ae_gaco, variable_label)

    table_rows.extend(table_result)
    rank_rows.extend(ranks_result)
    comparison_rows.append(comparison)
    detail_rows.extend(detail)


# write_csv("hasil_wilcoxon_test_statistics.csv", table_rows)
# write_csv("hasil_wilcoxon_ranks.csv", rank_rows)
# write_csv("hasil_wilcoxon_ringkasan_komparasi.csv", comparison_rows)
# write_csv("hasil_wilcoxon_detail_ae.csv", detail_rows)


# ============================================================
# Print hasil ke terminal
# ============================================================
print("===== HASIL UJI WILCOXON SIGNED-RANK TEST =====")
print("Hipotesis:")
print("H0: Tidak terdapat perbedaan signifikan antara hasil estimasi effort GA dan GACO.")
print("H1: Terdapat perbedaan signifikan antara hasil estimasi effort GA dan GACO.")
print("Kriteria keputusan:")
print("Asymp. Sig. < 0,05 maka H0 ditolak dan H1 diterima.")
print("Asymp. Sig. > 0,05 maka H0 gagal ditolak.")
print()

dataset_names = []
for row in rank_rows:
    if row["Dataset"] not in dataset_names:
        dataset_names.append(row["Dataset"])

for dataset_name in dataset_names:
    print(f"===== RANKS DATASET {dataset_name.upper()} =====")
    rank_table = [row for row in rank_rows if row["Dataset"] == dataset_name]
    print(format_table(rank_table, ["Variabel", "Ranks", "N", "Mean Rank", "Sum of Ranks"]))
    print()

print("===== TEST STATISTICS =====")
print(format_table(
    table_rows,
    [
        "Tabel",
        "Dataset",
        "Metode",
        "N",
        "Rata-rata AE / MAE",
        "Median AE",
        "Standar Deviasi AE",
        "Statistik Wilcoxon",
        "Asymp. Sig. (2-tailed)",
        "Keputusan",
    ]
))
print()

print("===== RINGKASAN KOMPARASI DATASET =====")
for row in comparison_rows:
    print(f"Dataset                         : {row['Dataset']}")
    print(f"Jumlah data                     : {row['Jumlah Data']}")
    print(f"MAE GA                          : {row['MAE GA']:.3f}")
    print(f"MAE GACO                        : {row['MAE GACO']:.3f}")
    print(f"Selisih MAE (GA - GACO)         : {row['Selisih MAE (GA - GACO)']:.3f}")
    print(f"Persentase penurunan MAE        : {row['Persentase Penurunan MAE (%)']:.3f}%")
    print(f"Jumlah proyek GACO lebih baik   : {row['Jumlah Proyek GACO Lebih Baik']}")
    print(f"Jumlah proyek GA lebih baik     : {row['Jumlah Proyek GA Lebih Baik']}")
    print(f"Jumlah proyek sama              : {row['Jumlah Proyek Sama']}")
    print(f"Statistik Wilcoxon              : {row['Statistik Wilcoxon']:.3f}")
    print(f"Asymp. Sig. (2-tailed)          : {row['Asymp. Sig. (2-tailed)']}")
    print(f"Keputusan                       : {row['Keputusan']}")
    print(f"Kesimpulan                      : {row['Kesimpulan']}")
    print(f"Arah perbandingan MAE           : {row['Arah Perbandingan MAE']}")
    print("-" * 80)

print()
print("File berhasil disimpan:")
print("- hasil_wilcoxon_test_statistics.csv")
print("- hasil_wilcoxon_ranks.csv")
print("- hasil_wilcoxon_ringkasan_komparasi.csv")
print("- hasil_wilcoxon_detail_ae.csv")
