import datetime
import unittest

import genetic


def get_fitness(genes):
    return genes.count(1)


def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}...{}\t{:3.2f}\t{}".format(
        ''.join(map(str, candidate.Genes[:15])),
        ''.join(map(str, candidate.Genes[-15:])),
        candidate.Fitness,
        timeDiff))


class OneMaxTests(unittest.TestCase):
    def test(self, length=100):
        geneset = [0, 1]
        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnGetFitness(genes):
            return get_fitness(genes)

        optimalFitness = length
        best = genetic.get_best(fnGetFitness, length, optimalFitness,
                                geneset, fnDisplay)
        self.assertEqual(best.Fitness, optimalFitness)

    def test_benchmark(self):
        genetic.Benchmark.run(lambda: self.test(500))


if __name__ == '__main__':
    unittest.main()
