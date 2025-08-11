import sys
import random
import statistics
import parameter as pr 
import datasetMaxwel as dt
import random_guessing as rg


class AlgoritmaGenetika:

    def __init__(self, parameterSetting):
        self.popsize = parameterSetting['popsize']
        self.crossoverRate = parameterSetting['crossoverRate']
        self.numOfDimension = parameterSetting['numOfDimension']
        self.mutationRate = parameterSetting['mutationRate']
        self.ranges = parameterSetting['ranges']
        self.maxIter = parameterSetting['maxIter']
        self.stoppingFitness = parameterSetting['stoppingFitness']

    def initialPopulasi(self):
        ''' INITIAL POPULASI '''
        chromosome = []
        for i in range(len(self.ranges)):
            lowBound = self.ranges[i][0]
            uppBound = self.ranges[i][1]
            chromosome.append(random.uniform(lowBound, uppBound))
        return chromosome

    def getDeceleration(self, chromosome):
        ''' DECELERATION '''
        hasilFf = chromosome[0] * chromosome[1] * chromosome[2] * chromosome[3] * chromosome[4] * chromosome[5] * chromosome[6] * chromosome[7] 
        hasilDf = chromosome[8] * chromosome[9] * chromosome[10] * \
            chromosome[11] * chromosome[12] * chromosome[13]* chromosome[14]* chromosome[15]* chromosome[16]* chromosome[17]* chromosome[18]* chromosome[19]* chromosome[20] * chromosome[21] 
        return hasilFf * hasilDf

    def calcAE(self, hasilDeceleration, vi, effort, actualEffort):
        ''' VALUE AE '''
        v = vi**hasilDeceleration
        hasilEffortInTime = effort / v
        hasilAE = abs(actualEffort - hasilEffortInTime)
        return hasilAE

    def selectCandidateChromosomes(self, probCummulatives, chromosomes):
        rets = []
        for i in range(len(probCummulatives)):
            # Langkah 1. Bangkitkan nilai acak [0,1]
            # =======================================
            randomValue = random.uniform(0, 1)
            # Langkah 2. Bandingkan nilai acak dengan nilai probabilitas kumulatif ke-i
            # ==========================================================================
            if randomValue > probCummulatives[i] and randomValue <= probCummulatives[i+1]:
                rets.append({'index': i, 'chromosome': chromosomes[i+1]})
        return rets

    def selectRoletteWheelChromosome(self, AEs, chromosomes):
        '''' CrossOver then selection with roda rolet '''
        # for that we should compute the cumulative probability values
        # that values finally should 1
        # ============================================================
        probCummulative = 0
        probCummulatives = []
        for AE in AEs:
            probability = AE / sum(AEs)
            probCummulative = probCummulative + probability
            probCummulatives.append(probCummulative)
        # then has value probality we can selection kandidat chromosome
        # =============================================================
        selectedCandidateChromosomes = self.selectCandidateChromosomes(
            probCummulatives, chromosomes)
        # a must has value then enough conditions for check for has value cannot empty!
        # =============================================================================
        while len(selectedCandidateChromosomes) == 0:
            selectedCandidateChromosomes = self.selectCandidateChromosomes(
                probCummulatives, chromosomes)
        return selectedCandidateChromosomes

    def generateRandomValues(self):
        rets = []
        # Generate random values along popsize
        # ====================================
        for i in range(self.popsize):
            if random.uniform(0, 1) < self.crossoverRate:
                rets.append(i)
        return rets

    def replaceChromosomesElement(self, chromosomes, chromosome, index):
        chromosomes[index] = chromosome
        return chromosomes

    def mainAlgen(self):
        AEs = []
        chromosomes = []
        selectedChromosomesToCrossover = []
        parentCandidatesIndex = []

        ''' INITIAL POPULASI AWAL '''
        # dataset
        # ================
        datas = dt.CetakDataset.maxwelDataset()
        iter = -1
        aeBestChromosomes = []
        estEffortBestChromosomes = []
        # loop dataset  ke-0....ke-...
        # ===================================
        for data in datas:
            iter +=1
            # print("Dataset ke-", iter + 1)
            for longPopsize in range(self.popsize):
                chromosome = AlgoritmaGenetika.initialPopulasi(self)
                chromosomes.append(chromosome)
                deceleration = self.getDeceleration(chromosome)
                vi = data[0]
                actEffort = data[2]
                effort = data[4]
                AE = self.calcAE(deceleration, vi, effort, actEffort)
                AEs.append(AE)
                chromosome = []


            ''' Seleksi chromosomes '''
            candidateNewChromosomes = self.selectRoletteWheelChromosome(
                AEs, chromosomes)

            ''' Create New Populas ke-i '''
            for candidateNewChromosome in candidateNewChromosomes:
                chromosomes = self.replaceChromosomesElement(
                    chromosomes, candidateNewChromosome['chromosome'], candidateNewChromosome['index'])

            ''' Update Population '''
            bestChromosomes = []
            for longMaxIter in range(self.maxIter):
                # Create random value then get indexs
                # important, this values we must more than 1 indexs
                # =================================================
                ''' Crossover '''
                randomIndexValues = self.generateRandomValues()
                while len(randomIndexValues) <= 1:
                    randomIndexValues = self.generateRandomValues()
                # selection chromosome for crossover
                # =================================
                selectedChromosomesToCrossover = []
                for randomIndexValue in randomIndexValues:
                    selectedChromosomesToCrossover.append(
                        {'chromosomes': chromosomes[randomIndexValue], 'index': randomIndexValue})
                # generate index parent pairs for crossover
                # =========================================
                parentCandidatesIndex = []
                for selectedChromosomesToCrossovers in selectedChromosomesToCrossover:
                    for selectedChromosomesToCrossoversed in selectedChromosomesToCrossover:
                        if selectedChromosomesToCrossovers['index'] != selectedChromosomesToCrossoversed['index']:
                            parentCandidatesIndex.append(
                                [selectedChromosomesToCrossovers['index'], selectedChromosomesToCrossoversed['index']])
                # form the index pair until it becomes unique
                # ===========================================
                sortedParentIndexes = []
                for parentIndex in parentCandidatesIndex:
                    parentIndex.sort()
                    sortedParentIndexes.append(parentIndex)
                finalParentIndexes = []
                for sortedParentIndex in sortedParentIndexes:
                    if sortedParentIndex not in finalParentIndexes:
                        finalParentIndexes.append(sortedParentIndex)
              
                ''' Create Offsets '''
                tempOffsets = []
                offsets = []
                for parentsIndex in finalParentIndexes:
                    cutPointIndex = random.randint(0, self.numOfDimension-1)
                    if cutPointIndex == self.numOfDimension-1:
                        for i in range(self.numOfDimension):
                            if i < self.numOfDimension-1:
                                tempOffsets.append(
                                    chromosomes[parentsIndex[1]][i])
                            else:
                                tempOffsets.append(
                                    chromosomes[parentsIndex[0]][cutPointIndex])
                    else:
                        for i in range(self.numOfDimension):
                            if i <= cutPointIndex:
                                tempOffsets.append(
                                    chromosomes[parentsIndex[0]][i])
                            else:
                                tempOffsets.append(
                                    chromosomes[parentsIndex[1]][i])
                    offsets.append(tempOffsets)
                    tempOffsets = []

                tempChromosomes = []
                tempAEs = []
                chromosomesOffsets = chromosomes + offsets
                # Count objektif value and fitness value from combined populations
                # ================================================================
                for chromosome in chromosomesOffsets:
                    deceleration = self.getDeceleration(chromosome)
                    AE = self.calcAE(deceleration, vi, effort, actEffort)
                    tempChromosomes.append(chromosome)
                    tempAEs.append(AE)
                # Sort the population in descending order by value
                # ================================================
                tempAEs.sort(reverse=False)
                chromosomes = []
                # create new populations
                # ======================
                for i in range(len(tempChromosomes)):
                    # memastikan jumlah kromosom sesuai ukuran populasi
                    # ================================================
                    if i <= self.popsize-1:
                        chromosomes.append(tempChromosomes[i])
                tempChromosomes = []
                ''' Process Mutasi '''
                # Count Amount mutation
                # =====================
                numOfMutation = round(self.mutationRate *
                                      (self.popsize * self.numOfDimension))
                for i in range(numOfMutation):
                    # select random index
                    # ===================
                    selectedChromosomeIndex = random.randint(0, self.popsize-1)
                    # select random gen in index population
                    # =====================================
                    selectedGenIndex = random.randint(0, self.numOfDimension-1)
                    # Change gen population termutation with random value sesuai rentang variabel designs
                    # ===================================================================================
                    mutatedChromosome = chromosomes[selectedChromosomeIndex]
                    lowerBound = self.ranges[selectedGenIndex][0]
                    upperBound = self.ranges[selectedGenIndex][1]
                    mutatedChromosome[selectedGenIndex] = random.uniform(
                        lowerBound, upperBound)
                    chromosomes[selectedChromosomeIndex] = mutatedChromosome
                # Count Fungsi Fitness / Value Objektif from new population
                # =========================================================
                tempAEs = []
                AEs = []
                results = []
                for chromosome in chromosomes:
                    deceleration = self.getDeceleration(chromosome)
                    v = vi**deceleration
                    estEffort = effort / v
                    AE = abs(actEffort - estEffort)
                    results.append([AE, chromosome, estEffort])
                bestChromosome = min(results)
                bestChromosomes.append(bestChromosome)
                
                # process a will stop if value chromosome terpenuhi with values stopingFitness
                # ============================================================================
                if bestChromosome[0] <= self.stoppingFitness:
                    break
                results = []
            bestChromosomes = min(bestChromosomes)
            # print("Best Chromosome:", bestChromosomes[1], "AE:", bestChromosomes[0], "estEffort:", bestChromosomes[2])
            # sys.exit()
            aeBestChromosomes.append(bestChromosomes[0])
            # print(f"{bestChromosomes[2]:.12f}".replace('.', ','))
            estEffortBestChromosomes.append(bestChromosomes[2])
        # process a count sum of the best chromosomes every datasets based on value AEs(value Objektif)
        # =============================================================================================
        sizeDataset = len(datas)
        averageBestChromosome = sum(aeBestChromosomes) / sizeDataset
        # print(f"{averageBestChromosome:.12f}".replace('.', ','))
        # for estEffort in estEffortBestChromosomes:
        #     print(f"{estEffort:.12f}".replace('.', ','))
        return {'MAE': averageBestChromosome, 'AEs': aeBestChromosomes, 'estEfforts': estEffortBestChromosomes}


# data, values rentang variabel design in prameter friction and dynamic factors
# =====================================================================
ranges = pr.prameterFfDf.parameter

# prameter setting
# ================
parameterSetting = {
    "popsize": 40,            # size populasi
    "crossoverRate": 0.25,
    "numOfDimension": len(ranges),    # long kromosome from prameter friction and dynamic factors
    "mutationRate": 1 / len(ranges),  # 1 / long kromosome
    "ranges": ranges,        # rentang variabel desain lowBound and UpperBound
    "maxIter": 60,           # Loop sebanyak value maxIter
    "stoppingFitness": 0.03
}

algen = AlgoritmaGenetika(parameterSetting)
hasil = algen.mainAlgen()
MAE = hasil['MAE']
estEfforts = hasil['estEfforts']
AE = hasil['AEs']
# print("MAE:", MAE)
# print("AEs:", AE)
# print('result estimasi : ', hasil)

# Evaluasi
# ========
runs = 1000
run = rg.RandomGuessing(estEfforts, runs)
randomGuessing = run.mainRandomGuessing()

MAE_P0 = randomGuessing['MAE_P0']
estEffortP0s = randomGuessing['estEffortP0s']
SA_P0 = 1 - (MAE / MAE_P0)

# a count standar deviasi. dari variabel ESTaePOs
# ===============================================
StDev_P0 = statistics.stdev(estEffortP0s)
ES = abs((MAE - MAE_P0) / StDev_P0)

# MBRE AND MIBRE
for estEffort in range(len(estEfforts)):
    minEstimated = min(estEfforts[estEffort],
                       randomGuessing['estEffortP0s'][estEffort])
    maxEstimated = max(estEfforts[estEffort],
                       randomGuessing['estEffortP0s'][estEffort])
    AE = abs(estEfforts[estEffort] - randomGuessing['estEffortP0s'][estEffort])
    aeMinEstimated = AE / minEstimated
    aeMaxEstimated = AE / maxEstimated
    # print('AE', AE, 'MAE_P0', MAE_P0, 'estEffortP0s', estEffortP0s, 'SA_P0', SA_P0, 'STDEV P0', StDev_P0,
    #       'ES', ES, 'AE MIN ESTIMATED', aeMinEstimated, 'AE MAX ESTIMATED', aeMaxEstimated)