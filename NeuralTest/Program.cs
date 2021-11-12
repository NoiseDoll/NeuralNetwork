using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetwork;

namespace NeuralTest
{
    class Program
    {
        static async Task Main(string[] args)
        {
            await TestIrisAsync();
        }

        private static async Task IsSevenTestAsync()
        {
            var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var ouput = new float[] { -1, -1, -1, -1, -1, -1, 1, -1, -1, -1 };
            var normalized = input.Select(i => i / 10f).ToArray();

            var perceptron = File.Exists("nn.json")
                ? await Perceptron.LoadAsync("nn.json")
                : Perceptron.CreateRandom(1, 1, 1, 4, new Random());

            for (int j = 0; j < 100000; j++)
            {
                var maxError = 0f;
                for (int i = 0; i < input.Length; i++)
                {
                    var x = normalized[i];
                    var y = ouput[i];

                    var error = perceptron.Sample(new[] { x }, new[] { y }, 0.1f, 0.05f, 0.1f, 1);
                    maxError = MathF.Max(maxError, error);
                }

                Console.WriteLine($"Iteration: {j}. Max error: {maxError}");

                if (maxError < 0.1f)
                {
                    break;
                }
            }

            await perceptron.SaveAsync("nn.json");

            foreach (var i in normalized)
            {
                var result = perceptron.Run(new[] { i });
                Console.WriteLine($"Result for {i}: {result[0]}");
            }

            Console.ReadLine();
        }

        private static async Task TestIrisAsync()
        {
            var data = await LoadIrisDataAsync("Data/iris.data");
            Normalize(data, out var dividers);

            var rng = new Random();
            var perceptron = File.Exists("nnIris.json")
                ? await Perceptron.LoadAsync("nnIris.json")
                : Perceptron.CreateRandom(4, 1, 2, 12, rng);

            var testData = Split(data, rng);

            Test(perceptron, testData, dividers);
            TrainBatch(perceptron, data);
            Test(perceptron, testData, dividers);

            static void TrainBatch(Perceptron perceptron, List<float[]> data)
            {
                perceptron.Batch(data.Select(d => d[..^1]), data.Select(d => new[] { d[^1] }), 0.1f, 10000, Log);
            }

            static void Train(Perceptron perceptron, List<float[]> data)
            {
                for (int j = 0; j < 100000; j++)
                {
                    var maxError = 0f;
                    for (int i = 0; i < data.Count; i++)
                    {
                        var line = data[i];
                        var input = line[..^1];
                        var output = line[^1];

                        var error = perceptron.Sample(input, new[] { output }, 0.1f, 0.05f, 0.1f, 1);
                        maxError = MathF.Max(maxError, error);
                    }

                    Console.WriteLine($"Iteration: {j}. Max error: {maxError}");

                    if (maxError < 0.1f)
                    {
                        break;
                    }
                }
            }

            static void Test(Perceptron perceptron, List<float[]> testData, float[] dividers)
            {
                var initialColor = Console.ForegroundColor;
                foreach (var line in testData)
                {
                    var input = line[..^1];
                    var output = line[^1];
                    var result = perceptron.Run(input);
                    var error = output - result[0];

                    var divider = dividers[^1];
                    var actual = result[0] * divider;
                    var expected = output * divider;

                    strMap.TryGetValue((int)MathF.Round(actual), out var actualName);
                    strMap.TryGetValue((int)MathF.Round(expected), out var expectedName);

                    Console.ForegroundColor = actualName == expectedName ? ConsoleColor.Green : ConsoleColor.Red;

                    input = Enumerable.Range(0, input.Length).Select((i) => input[i] * dividers[i]).ToArray();
                    Console.WriteLine($"Test error: {error:F2}");
                    Console.WriteLine($"Input: {string.Join(", ", input)}. Actual: {actualName}. Expected: {expectedName}");
                }

                Console.ForegroundColor = initialColor;
            }

            static void Log(int i, float e) => Console.WriteLine($"Iteration: {i}. Error: {e}");
        }

        private static async Task<List<float[]>> LoadIrisDataAsync(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new StreamReader(stream);

            var result = new List<float[]>();
            var maxVal = 0;
            string str = null;
            while (!string.IsNullOrWhiteSpace(str = await reader.ReadLineAsync()))
            {
                var splitted = str.Split(',');
                var floats = new float[splitted.Length];
                result.Add(floats);
                for (int i = 0; i < splitted.Length; i++)
                {
                    var t = splitted[i];
                    if (float.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
                    {
                        floats[i] = f;
                    }
                    else if (valueMap.TryGetValue(t, out var val))
                    {
                        floats[i] = val;
                    }
                    else
                    {
                        valueMap.Add(t, maxVal);
                        strMap.Add(maxVal, t);
                        floats[i] = maxVal;
                        maxVal++;
                    }
                }
            }

            return result;
        }

        private static void Normalize(List<float[]> values, out float[] dividers)
        {
            dividers = new float[values[0].Length];
            foreach (var line in values)
            {
                for (int i = 0; i < line.Length; i++)
                {
                    var v = line[i];
                    dividers[i] = MathF.Max(dividers[i], v);
                }
            }

            foreach (var line in values)
            {
                for (int i = 0; i < line.Length; i++)
                {
                    line[i] = line[i] / dividers[i];
                }
            }
        }

        private static List<float[]> Split(List<float[]> data, Random rng, float ratio = 0.3f)
        {
            var test = new List<float[]>((int)MathF.Ceiling(data.Count * ratio));
            for (int i = data.Count - 1; i >= 0; i--)
            {
                var random = rng.NextDouble();
                if (random < ratio)
                {
                    var line = data[i];
                    test.Add(line);
                    data.RemoveAt(i);
                }
            }

            return test;
        }

        private static readonly Dictionary<string, int> valueMap = new Dictionary<string, int>();
        private static readonly Dictionary<int, string> strMap = new Dictionary<int, string>();
    }
}
