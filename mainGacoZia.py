import sys
import math
import random
import statistics
import datasetZia as dataset
import ffdfZia as prameters
import random_guessing as rg

class HybridGACO:
    """
    GA + ACO:
    - GA backbone: roulette minimization, elitism, single-point crossover, gaussian mutation (clamped), early-stopping
    - ACO layer:
        * tau_parent[i]   : feromon untuk kecenderungan memilih kromosom-i sebagai parent
        * tau_cut[k]      : feromon untuk kecenderungan memilih cut-point k
      Seleksi parent & cut-point memakai ACS: eksploitasi (argmax) vs eksplorasi (roulette) dengan bobot pheromone^alpha * heuristic^beta
    """
    def __init__(self, parameterSetting):
        # --- GA params ---
        self.popsize         = parameterSetting['popsize']
        self.crossoverRate   = parameterSetting['crossoverRate']
        self.numOfDimension  = parameterSetting['numOfDimension']
        self.mutationRate    = parameterSetting['mutationRate']
        self.ranges          = parameterSetting['ranges']
        self.maxIter         = parameterSetting['maxIter']
        self.stoppingFitness = parameterSetting['stoppingFitness']
        self.patience        = parameterSetting.get('patience', 10)
        self.elite_k         = parameterSetting.get('elite_k', 1)
        self.mutationSigma   = parameterSetting.get('mutationSigma', 0.05)
        self.random_seed     = parameterSetting.get('seed', None)

        # --- ACO params ---
        self.rho      = parameterSetting.get('rho', 0.1)        # evaporasi
        self.alpha    = parameterSetting.get('alpha', 1.0)      # bobot pheromone
        self.beta     = parameterSetting.get('beta', 2.0)       # bobot heuristic
        self.q0       = parameterSetting.get('q0', 0.5)         # eksploitasi vs eksplorasi
        self.tau_init = parameterSetting.get('tau_init', 1.0)   # feromon awal
        self.tau_min  = parameterSetting.get('tau_min', 1e-6)
        self.tau_max  = parameterSetting.get('tau_max', 100.0)

        if self.random_seed is not None:
            random.seed(self.random_seed)

    # -------------------- UTIL GA --------------------
    def initialPopulasi(self):
        return [random.uniform(lb, ub) for (lb, ub) in self.ranges]

    def getDeceleration_ln(self, chromosome):
        s = 0.0
        for g in chromosome:
            g = max(g, 1e-12)
            s += math.log(g)
        return s  # = ln(D) dengan D = FR*DF

    def calcAE(self, lnD, vi, effort, actualEffort):
        if vi <= 0:
            return 1e18, 0.0
        # lnV = D*ln(vi) = exp(lnD)*ln(vi)
        lnV = math.exp(lnD) * math.log(vi)
        estEffort = effort * math.exp(-lnV)
        return abs(actualEffort - estEffort), estEffort

    def _fitness_from_AEs(self, AEs):
        fit = [1.0/(1.0 + ae) for ae in AEs]
        s = sum(fit)
        return [f/s if s>0 else 1.0/len(AEs) for f in fit]

    def _roulette(self, weights, k):
        # weights assumed normalized (sum ~ 1)
        import bisect
        cum = [0.0]
        c = 0.0
        for w in weights:
            c += w
            cum.append(c)
        if cum[-1] == 0.0:
            # fallback uniform
            n = len(weights)
            weights = [1.0/n]*n
            cum = [0.0]
            c = 0.0
            for w in weights:
                c += w
                cum.append(c)
        cum[-1] = 1.0
        idxs = []
        for _ in range(k):
            r = random.random()
            i = bisect.bisect_left(cum, r) - 1
            if i < 0: i = 0
            if i >= len(weights): i = len(weights)-1
            idxs.append(i)
        return idxs

    def _make_pairs(self, idxs):
        random.shuffle(idxs)
        return [(idxs[i], idxs[i+1]) for i in range(0, len(idxs)-1, 2)]

    def _single_point_crossover(self, p1, p2, cut):
        if self.numOfDimension == 1:
            return p1[:], p2[:]
        c1 = p1[:cut+1] + p2[cut+1:]
        c2 = p2[:cut+1] + p1[cut+1:]
        return c1, c2

    def _mutate_gene(self, val, low, high, sigma):
        nv = val + random.gauss(0.0, sigma*(high - low))
        return min(max(nv, low), high)

    def _mutate(self, chrom):
        for g in range(self.numOfDimension):
            if random.random() < self.mutationRate:
                lb, ub = self.ranges[g]
                chrom[g] = self._mutate_gene(chrom[g], lb, ub, self.mutationSigma)
        return chrom

    # -------------------- ACO: util & update --------------------
    def _aco_weights(self, tau, heuristic):
        # w_i ∝ (tau_i^alpha) * (heuristic_i^beta); normalisasi ke distribusi
        vals = []
        for t, h in zip(tau, heuristic):
            v = (max(t, self.tau_min) ** self.alpha) * (max(h, 1e-12) ** self.beta)
            vals.append(v)
        s = sum(vals)
        if s == 0:
            n = len(vals)
            return [1.0/n]*n
        return [v/s for v in vals]

    def _acs_choice(self, tau, heuristic):
        # ACS: dengan peluang q0 → pilih argmax; selain itu → roulette
        if random.random() < self.q0:
            # exploit
            scores = [(max(t, self.tau_min) ** self.alpha) * (max(h,1e-12) ** self.beta)
                      for t, h in zip(tau, heuristic)]
            return max(range(len(scores)), key=lambda i: scores[i])
        else:
            # explore
            weights = self._aco_weights(tau, heuristic)
            return self._roulette(weights, 1)[0]

    def _evaporate_all(self, arr):
        for i in range(len(arr)):
            arr[i] = max(self.tau_min, (1.0 - self.rho) * arr[i])

    def _deposit(self, arr, idx, delta):
        arr[idx] = min(self.tau_max, arr[idx] + delta)

    # -------------------- LOOP UTAMA GACO --------------------
    def mainAlgen(self):
        datas = dataset.CetakDataset.ziauddinDataset()  # [effort, vi, ..., actualEffort]
        aeBestChromosomes = []
        estEffortBestChromosomes = []

        for data in datas:
            effort, vi, actEffort = data[0], data[1], data[8]

            # init populasi
            chromosomes = [self.initialPopulasi() for _ in range(self.popsize)]
            # init feromon ACO
            tau_parent = [self.tau_init] * self.popsize          # per-kromosom
            tau_cut    = [self.tau_init] * self.numOfDimension   # per posisi cut

            # evaluasi awal
            scored = []
            for ch in chromosomes:
                lnD = self.getDeceleration_ln(ch)
                AE, est = self.calcAE(lnD, vi, effort, actEffort)
                scored.append((AE, ch, est))
            scored.sort(key=lambda x: x[0])

            best_AE, best_ch, best_est = scored[0]
            no_improve = 0

            # heuristik awal (untuk parent: berdasarkan fitness; untuk cut: pakai 1/numDim)
            AEs = [ae for ae, _, _ in scored]
            parent_heur = self._fitness_from_AEs(AEs)  # semakin tinggi → semakin dipilih
            cut_heur    = [1.0/self.numOfDimension]*self.numOfDimension

            for gen in range(self.maxIter):
                # --- elitism ---
                elites = [scored[i][1][:] for i in range(min(self.elite_k, len(scored)))]

                # --- ACO: pilih parent terarah pheromone+heuristik ---
                # kita pilih 2*floor(popsize/2) parent (dipasangkan 2-2)
                pair_count = (self.popsize // 2) * 2
                parent_idxs = []
                for _ in range(pair_count):
                    # choice via ACS memakai tau_parent & parent_heur
                    idx = self._acs_choice(tau_parent, parent_heur)
                    parent_idxs.append(idx)
                pairs = self._make_pairs(parent_idxs)

                # --- Reproduksi ---
                offspring = []
                for i, j in pairs:
                    p1 = chromosomes[i][:]
                    p2 = chromosomes[j][:]

                    # Probabilitas crossover global
                    if random.random() < self.crossoverRate:
                        # ACO: pilih cut-point via ACS menggunakan tau_cut & cut_heur
                        cut = self._acs_choice(tau_cut, cut_heur)
                        c1, c2 = self._single_point_crossover(p1, p2, cut)
                    else:
                        # tanpa crossover, keturunan = copy orang tua
                        c1, c2 = p1[:], p2[:]

                    # mutasi halus
                    offspring.append(self._mutate(c1))
                    offspring.append(self._mutate(c2))

                # --- Seleksi generasi berikutnya ---
                pool = chromosomes + offspring
                scored_pool = []
                for ch in pool:
                    lnD = self.getDeceleration_ln(ch)
                    AE, est = self.calcAE(lnD, vi, effort, actEffort)
                    scored_pool.append((AE, ch, est))
                scored_pool.sort(key=lambda x: x[0])

                # rebuild populasi: elit dulu, lalu terbaik dari pool
                new_pop = []
                for e in elites:
                    new_pop.append(e[:])
                for _, ch, _ in scored_pool:
                    if len(new_pop) >= self.popsize:
                        break
                    new_pop.append(ch[:])
                chromosomes = new_pop

                # evaluasi populasi baru
                scored = []
                for ch in chromosomes:
                    lnD = self.getDeceleration_ln(ch)
                    AE, est = self.calcAE(lnD, vi, effort, actEffort)
                    scored.append((AE, ch, est))
                scored.sort(key=lambda x: x[0])

                # track best + early stopping
                if scored[0][0] + 1e-12 < best_AE:
                    best_AE, best_ch, best_est = scored[0]
                    no_improve = 0
                else:
                    no_improve += 1

                if best_AE <= self.stoppingFitness or no_improve >= self.patience:
                    break

                # --- Update heuristik parent utk generasi berikutnya (berdasarkan AEs terbaru) ---
                AEs = [ae for ae, _, _ in scored]
                parent_heur = self._fitness_from_AEs(AEs)

                # --- ACO: evaporasi global ---
                self._evaporate_all(tau_parent)
                self._evaporate_all(tau_cut)

                # --- ACO: deposit pheromone dari solusi terbaik generasi ini ---
                # Perkiraan sederhana kontribusi: delta ~ 1/(1+best_AE)
                delta = 1.0/(1.0 + best_AE)

                # Deposit pada parent yang "baik": ambil top-2 individu (seolah parent terbaik)
                # Catatan: tanpa menyimpan pasangan terbaik real, pendekatan ini cukup efektif
                top2 = [scored[0][1], scored[1][1]] if len(scored) > 1 else [scored[0][1], scored[0][1]]
                # temukan index mereka di populasi saat ini
                idx_map = {id(chromosomes[i]): i for i in range(len(chromosomes))}
                for t in top2:
                    i = idx_map.get(id(t), None)
                    if i is not None:
                        self._deposit(tau_parent, i, delta)

                # Deposit pada cut-point yang “baik”: perkirakan cut yang memisah gen mayoritas elit
                # (heuristik ringan; jika ingin lebih akurat, simpan cut terpakai saat melahirkan anak terbaik)
                if self.numOfDimension > 1:
                    # pilih cut dengan index tengah sebagai fallback (bisa juga pilih acak berbobot tau_cut)
                    cut_idx = self._acs_choice(tau_cut, cut_heur)
                    self._deposit(tau_cut, cut_idx, delta)

            # simpan hasil per proyek
            aeBestChromosomes.append(best_AE)
            estEffortBestChromosomes.append(best_est)

        sizeDataset = len(aeBestChromosomes)
        MAE = sum(aeBestChromosomes)/sizeDataset if sizeDataset > 0 else float('inf')
        return {'MAE': MAE, 'AEs': aeBestChromosomes, 'estEfforts': estEffortBestChromosomes}


# -------------------- PARAMETER & EVALUASI --------------------
ranges = prameters.prameterFfDf.parameter

parameterSetting = {
    # GA
    "popsize": 40,
    "crossoverRate": 0.25,
    "numOfDimension": 13,
    "mutationRate": 0.05,      # lebih halus dari 1/13
    "ranges": ranges,
    "maxIter": 60,
    "stoppingFitness": 0.03,
    "patience": 10,
    "elite_k": 1,
    "mutationSigma": 0.05,
    "seed": 42,

    # ACO
    "rho": 0.1,         # evaporasi
    "alpha": 1.0,       # bobot pheromone
    "beta": 2.0,        # bobot heuristic (fitness)
    "q0": 0.5,          # eksploitasi vs eksplorasi
    "tau_init": 1.0,    # feromon awal
    "tau_min": 1e-6,
    "tau_max": 100.0
}

if __name__ == "__main__":
    gaco = HybridGACO(parameterSetting)
    hasil = gaco.mainAlgen()
    print('GACO result : ', hasil)

    # Baseline Random Guessing (tetap sama seperti versi Anda)
    MAE = hasil['MAE']
    estEfforts = hasil['estEfforts']
    runs = 1000
    run = rg.RandomGuessing(estEfforts, runs)
    randomGuessing = run.mainRandomGuessing()

    MAE_P0 = randomGuessing['MAE_P0']
    estEffortP0s = randomGuessing['estEffortP0s']
    SA_P0 = 1 - (MAE / MAE_P0)
    StDev_P0 = statistics.stdev(estEffortP0s)
    ES = abs((MAE - MAE_P0) / StDev_P0) if StDev_P0 > 0 else float('inf')

    for i in range(len(estEfforts)):
        minEstimated = min(estEfforts[i], estEffortP0s[i])
        maxEstimated = max(estEfforts[i], estEffortP0s[i])
        AE = abs(estEfforts[i] - estEffortP0s[i])
        aeMinEstimated = AE / (minEstimated if minEstimated != 0 else 1e-12)
        aeMaxEstimated = AE / (maxEstimated if maxEstimated != 0 else 1e-12)
        print('AE', AE, 'MAE_P0', MAE_P0, 'SA_P0', SA_P0, 'STDEV_P0', StDev_P0,
              'ES', ES, 'AE_MIN_EST', aeMinEstimated, 'AE_MAX_EST', aeMaxEstimated)