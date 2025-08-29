# gaco_init_maxwell.py
import math
import random
import statistics
import datasetMaxwel as dt      # pastikan modulmu sesuai
import parameter as pr          # berisi parameterFfDf.parameter (22 dim: 7 FF + 15 DF)
import random_guessing as rg

class HybridGA_InitACO_Maxwell:
    """
    Hybrid GA dengan ACO untuk INISIALISASI populasi awal.
    - Tahap awal: populasi dibangkitkan dengan ACO (guidance feromon + heuristik).
    - Tahap evolusi: murni GA (elitism, crossover, gaussian mutation, early-stopping).
    Evaluasi: AE per proyek → MAE keseluruhan.
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

        # --- ACO untuk inisialisasi ---
        self.rho      = parameterSetting.get('rho', 0.1)
        self.alpha    = parameterSetting.get('alpha', 1.0)
        self.beta     = parameterSetting.get('beta', 2.0)
        self.q0       = parameterSetting.get('q0', 0.5)
        self.tau_init = parameterSetting.get('tau_init', 1.0)
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

    def _lnD(self, chromosome):
        s = 0.0
        for g in chromosome:
            g = max(g, 1e-12)
            s += math.log(g)
        return s

    def _calc_AE(self, lnD, vi, effort, actualEffort):
        if vi <= 0:
            return 1e18, 0.0
        lnD_clamped = max(min(lnD, 50.0), -50.0)
        D = math.exp(lnD_clamped)
        lnV = D * math.log(vi)
        estEffort = effort * math.exp(-lnV)
        AE = abs(actualEffort - estEffort)
        return AE, estEffort

    def _evaluate(self, population, vi, effort, actual):
        scored = []
        for ch in population:
            lnD = self._lnD(ch)
            ae, est = self._calc_AE(lnD, vi, effort, actual)
            scored.append((ae, ch, est))
        scored.sort(key=lambda x: x[0])
        return scored

    def _fitness_from_AEs(self, AEs):
        fit = [1.0/(1.0 + ae) for ae in AEs]
        s = sum(fit)
        return [f/s if s > 0 else 1.0/len(fit) for f in fit]

    def _roulette(self, weights, k):
        import bisect
        cum = [0.0]; c = 0.0
        for w in weights:
            c += w; cum.append(c)
        if cum[-1] == 0.0:
            n = len(weights)
            weights = [1.0/n]*n; cum=[0.0]; c=0.0
            for w in weights: c+=w; cum.append(c)
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

    # =============== ACO hanya untuk inisialisasi ===============
    def _aco_initialize_population(self, vi, effort, actual, n_ants=30, n_iters=10, bins=10):
        import bisect
        edges, centers = [], []
        for (lb, ub) in self.ranges:
            step = (ub - lb) / bins
            e = [lb + i*step for i in range(bins+1)]
            c = [0.5*(e[i]+e[i+1]) for i in range(bins)]
            edges.append(e); centers.append(c)
        tau = [[self.tau_init]*bins for _ in range(self.numOfDimension)]

        pool = []
        for _ in range(n_iters):
            cands = []
            for _a in range(n_ants):
                ch = []
                for d in range(self.numOfDimension):
                    heuristic = [1.0/bins]*bins
                    if random.random() < self.q0:
                        scores = [max(tau[d][b], self.tau_min) ** self.alpha *
                                  max(heuristic[b], 1e-12) ** self.beta for b in range(bins)]
                        b_idx = max(range(bins), key=lambda i: scores[i])
                    else:
                        scores = [max(tau[d][b], self.tau_min) ** self.alpha *
                                  max(heuristic[b], 1e-12) ** self.beta for b in range(bins)]
                        s = sum(scores) or 1.0
                        r = random.random(); acc = 0.0; b_idx = 0
                        for i, sc in enumerate(scores):
                            acc += sc/s
                            if r <= acc: b_idx = i; break
                    ch.append(centers[d][b_idx])
                lnD = self._lnD(ch)
                ae, est = self._calc_AE(lnD, vi, effort, actual)
                cands.append((ae, ch, est))
            for d in range(self.numOfDimension):
                for b in range(bins):
                    tau[d][b] = max(self.tau_min, (1.0 - self.rho) * tau[d][b])
            cands.sort(key=lambda x: x[0])
            best_ae, best_ch, _ = cands[0]
            delta = 1.0/(1.0 + best_ae)
            for d, val in enumerate(best_ch):
                e = edges[d]
                b_idx = min(len(e)-2, max(0, bisect.bisect_right(e, val)-1))
                tau[d][b_idx] = min(self.tau_max, tau[d][b_idx] + delta)
            pool.extend(cands)

        pool.sort(key=lambda x: x[0])
        init_pop = [ch for _, ch, _ in pool[:self.popsize]]
        while len(init_pop) < self.popsize:
            init_pop.append(self._initial_chromosome())
        return init_pop

    # =============== MAIN LOOP ===============
    def run(self):
        rows = dt.CetakDataset.maxwelDataset()
        ae_results = []
        est_results = []
        actual_results = []

        for row in rows:
            vi     = row[self.vi_idx]
            actual = row[self.actual_idx]
            effort = row[self.effort_idx]

            # populasi awal pakai ACO
            population = self._aco_initialize_population(vi, effort, actual,
                                                         n_ants=30, n_iters=10, bins=10)
            scored = self._evaluate(population, vi, effort, actual)
            best_AE, best_ch, best_est = scored[0]
            no_improve = 0

            for gen in range(self.maxIter):
                elites = [scored[i][1][:] for i in range(min(self.elite_k, len(scored)))]
                AEs = [ae for ae, _, _ in scored]
                fitness = self._fitness_from_AEs(AEs)
                parent_idxs = self._roulette(fitness, (self.popsize // 2) * 2)
                pairs = self._make_pairs(parent_idxs)

                offspring = []
                for i, j in pairs:
                    p1 = population[i][:]
                    p2 = population[j][:]
                    if random.random() < self.crossoverRate:
                        cut = random.randint(0, self.numOfDimension-2)
                        c1, c2 = self._single_point_crossover(p1, p2, cut)
                    else:
                        c1, c2 = p1[:], p2[:]
                    offspring.append(self._mutate(c1))
                    offspring.append(self._mutate(c2))

                pool = population + offspring
                scored_pool = self._evaluate(pool, vi, effort, actual)
                new_pop = elites[:]
                for _, ch, _ in scored_pool:
                    if len(new_pop) >= self.popsize:
                        break
                    new_pop.append(ch[:])

                population = new_pop
                scored = self._evaluate(population, vi, effort, actual)

                if scored[0][0] + 1e-12 < best_AE:
                    best_AE, best_ch, best_est = scored[0]
                    no_improve = 0
                else:
                    no_improve += 1
                if best_AE <= self.stoppingFitness or no_improve >= self.patience:
                    break

            ae_results.append(best_AE)
            est_results.append(best_est)
            actual_results.append(actual)

        MAE = sum(ae_results)/len(ae_results) if ae_results else float('inf')
        return {'MAE': MAE, 'AEs': ae_results, 'estEfforts': est_results, 'actualEfforts': actual_results}


# ========================= PARAMETER & RUN =========================
if __name__ == '__main__':
    ranges = pr.prameterFfDf.parameter   # 22 rentang (7 FF + 15 DF)

    parameterSetting = {
        # GA
        "popsize": 40,
        "crossoverRate": 0.7,
        "numOfDimension": len(ranges),
        "mutationRate": 0.05,
        "ranges": ranges,
        "maxIter": 60,
        "stoppingFitness": 0.03,
        "patience": 10,
        "elite_k": 1,
        "mutationSigma": 0.05,
        "seed": 42,

        # ACO init
        "rho": 0.1,
        "alpha": 1.0,
        "beta": 2.0,
        "q0": 0.5,
        "tau_init": 1.0,
        "tau_min": 1e-6,
        "tau_max": 100.0,

        # Mapping kolom dataset
        "vi_idx": 0,
        "actual_idx": 2,
        "effort_idx": 4,
    }

    gaco = HybridGA_InitACO_Maxwell(parameterSetting)
    result = gaco.run()
    print('Hybrid Init-ACO GA (Maxwell):', result)

    # ==== Evaluasi baseline (contoh random guessing) ====
    MAE = result['MAE']
    estEfforts = result['estEfforts']
    actualEfforts = result['actualEfforts']

    runs = 1000
    run = rg.RandomGuessing(actualEfforts, runs)
    randomGuessing = run.mainRandomGuessing()

    MAE_P0 = randomGuessing['MAE_P0_mean']
    SA = 1 - (MAE / MAE_P0)

    StDev_P0 = statistics.stdev(randomGuessing['MAE_P0_samples'])
    ES = (MAE_P0 - MAE) / StDev_P0 if StDev_P0 > 0 else float('inf')

    # Hitung MBRE/MIBRE
    def mbre(y, yhat): return abs(y - yhat) / (min(y, yhat) if min(y, yhat) != 0 else 1e-12)
    def mibre(y, yhat): return abs(y - yhat) / (max(y, yhat) if max(y, yhat) != 0 else 1e-12)
    MBRE = sum(mbre(a, e) for a, e in zip(actualEfforts, estEfforts)) / len(actualEfforts)
    MIBRE = sum(mibre(a, e) for a, e in zip(actualEfforts, estEfforts)) / len(actualEfforts)

    print({
        'MAE_GA+ACOinit': MAE,
        'SA': SA,
        'ES': ES,
        'MBRE': MBRE,
        'MIBRE': MIBRE,
        'MAE_P0': MAE_P0
    })