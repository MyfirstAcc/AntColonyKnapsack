using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AntColony
{
    class AntColonyOptimization
    {
        private static Random _random = new Random();
        private double[] _pheromones;
        private int[] _weights;
        private int[] _values;
        private int _capacity;
        private double _alpha;
        private double _beta;
        private double _rho;
        private double _q;
        private int _iterations;
        private int _ants;
        private int _threads;

        public AntColonyOptimization(double[] pheromones, int[] weights, int[] values, int capacity, double alpha, double beta, double rho, double q, int iterations, int ants, int threads)
        {
            _pheromones = pheromones;
            _weights = weights;
            _values = values;
            _capacity = capacity;
            _alpha = alpha;
            _beta = beta;
            _rho = rho;
            _q = q;
            _iterations = iterations;
            _ants = ants;
            _threads = threads;
        }

        private static int SelectItem(double[] probabilities)
        {
            double sum = probabilities.Sum();
            double rand = _random.NextDouble() * sum;
            double cumulative = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (rand <= cumulative)
                    return i;
            }
            return probabilities.Length - 1;
        }

        private List<int> AntSolution()
        {
            int numItems = _weights.Length;
            List<int> selectedItems = new List<int>();
            int currentWeight = 0;
            List<int> availableItems = Enumerable.Range(0, numItems).ToList();

            while (availableItems.Count > 0)
            {
                double[] heuristic = availableItems
                    .Select(i => _values[i] / (_weights[i] + 1e-10))
                    .ToArray();
                double[] probabilities = availableItems
                    .Select(i => Math.Pow(_pheromones[i], _alpha) * Math.Pow(heuristic[availableItems.IndexOf(i)], _beta))
                    .ToArray();

                double sum = probabilities.Sum();
                for (int i = 0; i < probabilities.Length; i++)
                    probabilities[i] /= sum;

                int selectedIndex = SelectItem(probabilities);
                int item = availableItems[selectedIndex];
                availableItems.RemoveAt(selectedIndex);

                if (currentWeight + _weights[item] <= _capacity)
                {
                    selectedItems.Add(item);
                    currentWeight += _weights[item];
                }
            }
            return selectedItems;
        }

        public (List<int>, int) Solve()
        {
            List<int> bestSolution = null;
            int bestValue = int.MinValue;

            for (int i = 0; i < _iterations; i++)
            {
                double[] localPheromones = new double[_weights.Length];
                object lockObject = new object();

                Parallel.For(0, _threads, _ =>
                {
                    double[] localGroupPheromones = new double[_weights.Length];
                    List<int> localBestSolution = null;
                    int localBestValue = int.MinValue;
                    int antsPerGroup = _ants / _threads;

                    for (int j = 0; j < antsPerGroup; j++)
                    {
                        List<int> solution = AntSolution();
                        int value = solution.Sum(index => _values[index]);
                        int weight = solution.Sum(index => _weights[index]);

                        if (value > localBestValue && weight <= _capacity)
                        {
                            localBestSolution = new List<int>(solution);
                            localBestValue = value;
                        }
                        foreach (var item in solution)
                        {
                            localGroupPheromones[item] += _q / value;
                        }
                    }
                    Console.WriteLine($"Поток {Task.CurrentId} - Решение группы: {localBestValue}");

                    lock (lockObject) // критическая секция
                    {
                        if (localBestValue > bestValue) // разделяемый ресурс - лучшее значение 
                        {
                            bestSolution = localBestSolution;
                            bestValue = localBestValue;
                        }
                        for (int k = 0; k < _weights.Length; k++)
                        {
                            localPheromones[k] += localGroupPheromones[k]; // обновление локального феромона
                        }
                    }
                });

                for (int k = 0; k < _weights.Length; k++)
                {
                    _pheromones[k] = (1 - _rho) * _pheromones[k] + localPheromones[k]; // глобальное обновление феромона
                }
                Console.WriteLine($"--- Итерация {i + 1}, Лучшее значение: {bestValue}");
            }
            return (bestSolution, bestValue);
        }

        public static (int[], int[]) GenerateModel(int n, (int, int) rangeV, (int, int) rangeW)
        {
            int[] values = new int[n];
            int[] weights = new int[n];
            for (int i = 0; i < n; i++)
            {
                values[i] = _random.Next(rangeV.Item1, rangeV.Item2);
                weights[i] = _random.Next(rangeW.Item1, rangeW.Item2);
            }
            return (values, weights);
        }

        static void Main()
        {
            int n = 1000;
            var (values, weights) = GenerateModel(n, (100, 500), (30, 100));
            int capacity = 500;
            int ants = 100;
            int threads = 16;
            double alpha = 1.0;
            double beta = 2.0;
            double rho = 0.5;
            double q = 100;
            int iterations = 20;
            double[] pheromones = Enumerable.Repeat(1.0, weights.Length).ToArray();

            var aco = new AntColonyOptimization(pheromones, weights, values, capacity, alpha, beta, rho, q, iterations, ants, threads);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var (bestSolution, bestValue) = aco.Solve();
            watch.Stop();

            Console.WriteLine("Лучшее решение: [ " + string.Join(", ", bestSolution) + " ]");
            Console.WriteLine($"Лучшее значение: {bestValue}");
            Console.WriteLine($"Время выполнения: {watch.ElapsedMilliseconds / 1000.0:F2} секунд");
        }
    }
}
