# gaco_maxwell.py
import math
import random
import statistics
import datasetMaxwel as dt      # pastikan modulmu sesuai
import parameter as pr          # berisi prameterFfDf.parameter (22 dim: 7 FF + 15 DF)
import random_guessing as rg

class HybridGACO_Maxwell:
    """
    GA + ACO untuk dataset Maxwell (22 dimensi: 7 FF + 15 DF)
    - GA backbone: elitism, single-point crossover, gaussian mutation (clamp), early-stopping
    - ACO layer:
        * tau_parent[i] : feromon memilih individu ke-i sebagai parent
        * tau_cut[k]    : feromon memilih cut-point k
      Seleksi parent & cut memakai ACS (q0 eksploitasi, selain itu eksplorasi lewat roulette).
    - Stabilitas numerik: hitung di log-space.
    """
    def __init__(self, parameterSetting):
        # --- GA ---
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
        self.seed            = parameterSetting.get('seed', None)

        # --- ACO ---
        self.rho      = parameterSetting.get('rho', 0.1)       # evaporasi
        self.alpha    = parameterSetting.get('alpha', 1.0)     # bobot pheromone
        self.beta     = parameterSetting.get('beta', 2.0)      # bobot heuristic
        self.q0       = parameterSetting.get('q0', 0.5)        # eksploitasi vs eksplorasi
        self.tau_init = parameterSetting.get('tau_init', 1.0)  # feromon awal
        self.tau_min  = parameterSetting.get('tau_min', 1e-6)
        self.tau_max  = parameterSetting.get('tau_max', 100.0)

        # --- Mapping kolom Maxwell (bisa override via parameterSetting) ---
        self.vi_idx     = parameterSetting.get('vi_idx', 0)
        self.actual_idx = parameterSetting.get('actual_idx', 2)
        self.effort_idx = parameterSetting.get('effort_idx', 4)

        if self.seed is not None:
            random.seed(self.seed)

    # =============== Util dasar GA ===============
    def _initial_chromosome(self):
        return [random.uniform(lb, ub) for (lb, ub) in self.ranges]

    def _initial_population(self):
        return [self._initial_chromosome() for _ in range(self.popsize)]

    def _lnD(self, chromosome):
        # ln(D) = sum ln(gen); clamp gen > 0
        s = 0.0
        for g in chromosome:
            g = max(g, 1e-12)
            s += math.log(g)
        return s

    def _calc_AE(self, lnD, vi, effort, actualEffort):
        # v = vi ** D; ln v = D * ln vi = exp(lnD) * ln vi
        if vi <= 0:
            return 1e18, 0.0
        # clamp lnD untuk keamanan numerik
        lnD_clamped = max(min(lnD, 50.0), -50.0)    # opsional
        D = math.exp(lnD_clamped)
        lnV = D * math.log(vi)
        # effort / v = effort * exp(-lnV)
        estEffort = effort * math.exp(-lnV)
        AE = abs(actualEffort - estEffort)
        return AE, estEffort

    def _evaluate(self, population, vi, effort, actual):
        scored = []
        for ch in population:
            lnD = self._lnD(ch)
            ae, est = self._calc_AE(lnD, vi, effort, actual)
            scored.append((ae, ch, est))
        scored.sort(key=lambda x: x[0])  # AE ascending
        return scored

    def _fitness_from_AEs(self, AEs):
        fit = [1.0/(1.0 + ae) for ae in AEs]
        s = sum(fit)
        return [f/s if s > 0 else 1.0/len(fit) for f in fit]

    def _roulette(self, weights, k):
        # weights assume normalized
        import bisect
        cum = [0.0]
        c = 0.0
        for w in weights:
            c += w
            cum.append(c)
        if cum[-1] == 0.0:
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
        # cut di [0 .. numDim-2] supaya ada bagian kiri & kanan
        cut = max(0, min(cut, self.numOfDimension - 2))
        c1 = p1[:cut+1] + p2[cut+1:]
        c2 = p2[:cut+1] + p1[cut+1:]
        return c1, c2

    def _mutate(self, ch):
        for i in range(self.numOfDimension):
            if random.random() < self.mutationRate:
                lb, ub = self.ranges[i]
                step = self.mutationSigma * (ub - lb)
                ch[i] = max(lb, min(ub, ch[i] + random.gauss(0.0, step)))
        return ch

    # =============== ACO util ===============
    def _aco_weights(self, tau, heuristic):
        vals = []
        for t, h in zip(tau, heuristic):
            v = (max(t, self.tau_min) ** self.alpha) * (max(h, 1e-12) ** self.beta)
            vals.append(v)
        s = sum(vals)
        return [v/s if s > 0 else 1.0/len(vals) for v in vals]

    def _acs_choice(self, tau, heuristic):
        if random.random() < self.q0:
            # exploit: argmax
            scores = [(max(t, self.tau_min) ** self.alpha) * (max(h, 1e-12) ** self.beta)
                      for t, h in zip(tau, heuristic)]
            return max(range(len(scores)), key=lambda i: scores[i])
        else:
            # explore: roulette
            weights = self._aco_weights(tau, heuristic)
            return self._roulette(weights, 1)[0]

    def _evaporate_all(self, arr):
        for i in range(len(arr)):
            arr[i] = max(self.tau_min, (1.0 - self.rho) * arr[i])

    def _deposit(self, arr, idx, delta):
        arr[idx] = min(self.tau_max, arr[idx] + delta)

    # =============== MAIN LOOP ===============
    def run(self):
        rows = dt.CetakDataset.maxwelDataset()
        ae_results = []
        est_results = []

        for row in rows:
            vi     = row[self.vi_idx]
            actual = row[self.actual_idx]
            effort = row[self.effort_idx]

            # inisialisasi pop & pheromone PER PROYEK (reset!)
            population = self._initial_population()
            tau_parent = [self.tau_init] * self.popsize
            # cut-point di antara gen: 0..numDim-2
            tau_cut    = [self.tau_init] * max(1, self.numOfDimension - 1)

            # evaluasi awal
            scored = self._evaluate(population, vi, effort, actual)
            best_AE, best_ch, best_est = scored[0]
            no_improve = 0

            # heuristik awal
            AEs = [ae for ae, _, _ in scored]
            parent_heur = self._fitness_from_AEs(AEs)
            cut_heur    = [1.0 / len(tau_cut)] * len(tau_cut)

            for gen in range(self.maxIter):
                # --- elitism ---
                elites = [scored[i][1][:] for i in range(min(self.elite_k, len(scored)))]

                # --- pilih parent via ACO (ACS) ---
                pair_count = (self.popsize // 2) * 2
                parent_idxs = [self._acs_choice(tau_parent, parent_heur) for _ in range(pair_count)]
                pairs = self._make_pairs(parent_idxs)

                # --- crossover & mutasi (track cut histogram) ---
                cut_counts = [0] * len(tau_cut)
                offspring = []
                for i, j in pairs:
                    p1 = population[i][:]
                    p2 = population[j][:]
                    if random.random() < self.crossoverRate and len(tau_cut) > 0:
                        cut = self._acs_choice(tau_cut, cut_heur)
                        cut_counts[cut] += 1
                        c1, c2 = self._single_point_crossover(p1, p2, cut)
                    else:
                        c1, c2 = p1[:], p2[:]
                    offspring.append(self._mutate(c1))
                    offspring.append(self._mutate(c2))

                # --- seleksi generasi berikutnya ---
                pool = population + offspring
                scored_pool = self._evaluate(pool, vi, effort, actual)

                # rebuild populasi: elit + terbaik dari pool
                new_pop = []
                for e in elites:
                    new_pop.append(e[:])
                for _, ch, _ in scored_pool:
                    if len(new_pop) >= self.popsize:
                        break
                    new_pop.append(ch[:])

                population = new_pop
                scored = self._evaluate(population, vi, effort, actual)

                # --- track best & early-stopping ---
                if scored[0][0] + 1e-12 < best_AE:
                    best_AE, best_ch, best_est = scored[0]
                    no_improve = 0
                else:
                    no_improve += 1

                if best_AE <= self.stoppingFitness or no_improve >= self.patience:
                    break

                # --- update heuristik parent utk next gen ---
                AEs = [ae for ae, _, _ in scored]
                parent_heur = self._fitness_from_AEs(AEs)

                # --- evaporasi pheromone ---
                self._evaporate_all(tau_parent)
                self._evaporate_all(tau_cut)

                # --- deposit pheromone ---
                delta = 1.0 / (1.0 + best_AE)

                # deposit ke parent: top-2 individu generasi ini
                idx_map = {id(population[i]): i for i in range(len(population))}
                top2 = [scored[0][1], scored[1][1] if len(scored) > 1 else scored[0][1]]
                for t in top2:
                    i = idx_map.get(id(t), None)
                    if i is not None:
                        self._deposit(tau_parent, i, delta)

                # deposit ke cut-point: cut yang paling sering dipakai
                if len(cut_counts) > 0:
                    best_cut = max(range(len(cut_counts)), key=lambda k: cut_counts[k])
                    self._deposit(tau_cut, best_cut, delta)

            # simpan hasil per proyek
            ae_results.append(best_AE)
            est_results.append(best_est)

        MAE = sum(ae_results)/len(ae_results) if ae_results else float('inf')
        return {'MAE': MAE, 'AEs': ae_results, 'estEfforts': est_results}


# ========================= PARAMETER & RUN =========================
if __name__ == '__main__':
    ranges = pr.prameterFfDf.parameter   # 22 rentang (7 FF + 15 DF)

    parameterSetting = {
        # GA
        "popsize": 40,
        "crossoverRate": 0.7,
        "numOfDimension": len(ranges),
        "mutationRate": 0.05,         # lebih halus dari 1/len(ranges)
        "ranges": ranges,
        "maxIter": 60,
        "stoppingFitness": 0.03,
        "patience": 10,
        "elite_k": 1,
        "mutationSigma": 0.05,
        "seed": 42,

        # ACO
        "rho": 0.1,
        "alpha": 1.0,
        "beta": 2.0,
        "q0": 0.5,
        "tau_init": 1.0,
        "tau_min": 1e-6,
        "tau_max": 100.0,

        # Mapping kolom (ubah kalau dataset berbeda urutan kolomnya)
        "vi_idx": 0,
        "actual_idx": 2,
        "effort_idx": 4,
    }

    gaco = HybridGACO_Maxwell(parameterSetting)
    result = gaco.run()
    print('GACO Maxwell result:', result)

    # ==== Evaluasi baseline (tetap seperti kodenya) ====
    MAE = result['MAE']
    estEfforts = result['estEfforts']

    runs = 1000
    run = rg.RandomGuessing(estEfforts, runs)
    randomGuessing = run.mainRandomGuessing()

    MAE_P0 = randomGuessing['MAE_P0']
    estEffortP0s = randomGuessing['estEffortP0s']
    SA_P0 = 1 - (MAE / MAE_P0)

    StDev_P0 = statistics.stdev(estEffortP0s)
    ES = abs((MAE - MAE_P0) / StDev_P0) if StDev_P0 > 0 else float('inf')

    # MBRE/MIBRE per proyek
    for i in range(len(estEfforts)):
        minEstimated = min(estEfforts[i], estEffortP0s[i])
        maxEstimated = max(estEfforts[i], estEffortP0s[i])
        AE = abs(estEfforts[i] - estEffortP0s[i])
        aeMinEstimated = AE / (minEstimated if minEstimated != 0 else 1e-12)
        aeMaxEstimated = AE / (maxEstimated if maxEstimated != 0 else 1e-12)
        # print atau simpan sesuai kebutuhan
        # print('AE', AE, 'MAE_P0', MAE_P0, 'SA_P0', SA_P0, 'STDEV_P0', StDev_P0,
        #       'ES', ES, 'AE_MIN_EST', aeMinEstimated, 'AE_MAX_EST', aeMaxEstimated)