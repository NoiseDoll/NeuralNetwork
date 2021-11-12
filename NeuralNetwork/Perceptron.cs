using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Perceptron
    {
        public List<Layer> Layers { get; set; }

        [JsonIgnore]
        public List<TrainedLayer> Trainers { get; set; }

        public static Perceptron CreateRandom(int inputCount, int outputCount, int layerCount, int neuronCount, Random rng)
        {
            var layers = new List<Layer>(layerCount + 1);
            var trainers = new List<TrainedLayer>(layerCount + 1);
            var perceptron = new Perceptron
            {
                Layers = layers,
                Trainers = trainers,
            };

            var currentCount = inputCount;
            for (int i = 0; i < layerCount; i++)
            {
                var layer = Layer.CreateRandom(currentCount, neuronCount, rng);
                layers.Add(layer);
                currentCount = neuronCount;
            }

            var lastLayer = Layer.CreateRandom(currentCount, outputCount, rng);
            layers.Add(lastLayer);

            currentCount = inputCount;
            for (int i = 0; i < layerCount; i++)
            {
                var layer = TrainedLayer.Create(currentCount, neuronCount);
                trainers.Add(layer);
                currentCount = neuronCount;
            }

            var lastTrainer = TrainedLayer.Create(currentCount, outputCount);
            trainers.Add(lastTrainer);

            return perceptron;
        }

        public static async Task<Perceptron> LoadAsync(string path, CancellationToken cancellationToken = default)
        {
            using var stream = File.OpenRead(path);
            var perceptron = await JsonSerializer.DeserializeAsync<Perceptron>(stream, null, cancellationToken);

            foreach (var layer in perceptron.Layers)
            {
                layer.Output = new float[layer.Neurons.Length];
            }

            var layersCount = perceptron.Layers.Count;
            var trainers = new List<TrainedLayer>(layersCount);
            for (int i = 0; i < layersCount; i++)
            {
                var layer = perceptron.Layers[i];
                var inputCount = layer.Neurons[0].Weights.Length;
                var trainer = TrainedLayer.Create(inputCount, layer.Neurons.Length);
                trainers.Add(trainer);
            }

            perceptron.Trainers = trainers;
            return perceptron;
        }

        public async Task SaveAsync(string path, CancellationToken cancellationToken = default)
        {
            using var stream = File.Create(path);
            var options = new JsonSerializerOptions { WriteIndented = true, };
            await JsonSerializer.SerializeAsync(stream, this, options, cancellationToken);
        }

        public float[] Run(float[] input)
        {
            var current = input;
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                ComputeOutputSimulate(current, layer);
                current = layer.Output;
            }

            return current.ToArray();
        }

        public float Sample(
            float[] input,
            float[] output,
            float learningRate = 0.1f,
            float momentum = 0f,
            float errorTolerance = 0f,
            int maxIterations = 1)
        {
            var error = 0f;
            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                PropagateForward(input);

                error = GetError(output);
                if (error < errorTolerance)
                {
                    return error;
                }

                PropagateBackward(output);

                var current = input;
                for (int i = 0; i < Layers.Count; i++)
                {
                    var layer = Layers[i];
                    var trainer = Trainers[i];

                    for (int j = 0; j < layer.Neurons.Length; j++)
                    {
                        var neuron = layer.Neurons[j];
                        var gradient = trainer.Gradients[j];
                        var delta = trainer.Deltas[j];

                        {
                            var d = learningRate * gradient + delta.Bias * momentum;
                            neuron.Bias += d;
                            delta.Bias = d;
                        }

                        var weights = neuron.Weights;
                        var deltaWeights = delta.Weights;
                        for (int k = 0; k < weights.Length; k++)
                        {
                            var d = learningRate * gradient * current[k] + deltaWeights[k] * momentum;
                            weights[k] += d;
                            deltaWeights[k] = d;
                        }
                    }

                    current = layer.Output;
                }
            }

            return error;
        }

        public float Batch(
            IEnumerable<float[]> inputs, IEnumerable<float[]> outputs, float errorTolerance, int maxIterations, Action<int, float> iterationCallback)
        {
            var weightCount = GetWeightCount();
            var oldGradients = new float[weightCount];
            var newGradients = new float[weightCount];
            var magnitude = Enumerable.Repeat(0.1f, weightCount).ToArray();

            var error = 0f;
            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                error = 0f;
                var k = 0;
                var count = 0;
                foreach (var (input, output) in inputs.Zip(outputs))
                {
                    PropagateForward(input);
                    error += GetError(output);
                    PropagateBackward(output);

                    var current = input;
                    k = 0;
                    count++;

                    for (int i = 0; i < Layers.Count; i++)
                    {
                        var trainer = Trainers[i];
                        foreach (var gradient in trainer.Gradients)
                        {
                            newGradients[k++] += gradient;
                            foreach (var signal in current)
                            {
                                newGradients[k++] += gradient * signal;
                            }
                        }

                        var layer = Layers[i];
                        current = layer.Output;
                    }
                }

                error /= count;
                if (error < errorTolerance)
                {
                    return error;
                }

                iterationCallback?.Invoke(iteration, error);

                k = 0;
                foreach (var layer in Layers)
                {
                    foreach (var neuron in layer.Neurons)
                    {
                        neuron.Bias = UpdateWeight(
                            ref newGradients[k], ref oldGradients[k], ref magnitude[k], neuron.Bias);
                        k++;

                        for (int i = 0; i < neuron.Weights.Length; i++, k++)
                        {
                            var weight = neuron.Weights[i];
                            neuron.Weights[i] = UpdateWeight(
                                ref newGradients[k], ref oldGradients[k], ref magnitude[k], weight);
                        }
                    }
                }
            }

            return error;
        }

        private void PropagateForward(float[] inputs)
        {
            var current = inputs;
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var trainer = Trainers[i];
                ComputeOutputTrain(current, layer, trainer);
                current = layer.Output;
            }
        }

        private void PropagateBackward(float[] output)
        {
            {
                var last = Layers.Count - 1;
                var results = Layers[last].Output;
                var lastTrainer = Trainers[last];
                var derivatives = lastTrainer.Derivatives;
                var gradients = lastTrainer.Gradients;

                for (int i = 0; i < output.Length; i++)
                {
                    gradients[i] = (output[i] - results[i]) * derivatives[i];
                }
            }

            for (int i = Layers.Count - 1; i >= 1; i--)
            {
                var neurons = Layers[i].Neurons;
                var gradients = Trainers[i].Gradients;
                var previousTrainer = Trainers[i - 1];
                var previousDerivatives = previousTrainer.Derivatives;
                var previousGradients = previousTrainer.Gradients;

                for (int j = 0; j < previousGradients.Length; j++)
                {
                    var delta = 0f;
                    for (int k = 0; k < gradients.Length; k++)
                    {
                        var gradient = gradients[k];
                        var weight = neurons[k].Weights[j];
                        delta += gradient * weight;
                    }

                    previousGradients[j] = delta * previousDerivatives[j];
                }
            }
        }

        private float GetError(float[] reference)
        {
            var error = 0f;
            var output = Layers[Layers.Count - 1].Output;
            for (int i = 0; i < reference.Length; i++)
            {
                var delta = reference[i] - output[i];
                error += MathF.Sqrt(delta * delta);
            }

            return error / reference.Length;
        }

        private int GetWeightCount()
        {
            var count = 0;
            foreach (var layer in Layers)
            {
                count += layer.Output.Length * (layer.Neurons[0].Weights.Length + 1);
            }

            return count;
        }

        private static void ComputeOutputSimulate(float[] input, Layer layer)
        {
            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                var neuron = layer.Neurons[i];
                var sum = NetSum(input, neuron);
                var output = Sigmoid(sum);
                layer.Output[i] = output;
            }
        }

        private static void ComputeOutputTrain(float[] input, Layer layer, TrainedLayer trainer)
        {
            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                var neuron = layer.Neurons[i];
                var sum = NetSum(input, neuron);
                var output = Sigmoid(sum);
                layer.Output[i] = output;
                trainer.Derivatives[i] = DerivativeSigmoid(output);
            }
        }

        private static float NetSum(float[] input, Neuron neuron)
        {
            var sum = neuron.Bias;
            for (int i = 0; i < neuron.Weights.Length; i++)
            {
                sum += input[i] * neuron.Weights[i];
            }

            return sum;
        }

        private static float UpdateWeight(
            ref float newGradient, ref float oldGradient, ref float magnitude, float weight)
        {
            const float MaxMagnitude = 50f;
            const float MinMagnitude = 1e-6f;
            const float nMinus = 0.5f;
            const float nPlus = 1.2f;

            var mult = oldGradient * newGradient;
            if (mult > 0f)
            {
                magnitude = MathF.Min(magnitude * nPlus, MaxMagnitude);
                var delta = MathF.Sign(newGradient) * magnitude;
                weight += delta;
                oldGradient = newGradient;
            }
            else if (mult < 0f)
            {
                magnitude = MathF.Max(magnitude * nMinus, MinMagnitude);
                oldGradient = 0f;
            }
            else
            {
                var delta = MathF.Sign(newGradient) * magnitude;
                weight += delta;
                oldGradient = newGradient;
            }

            newGradient = 0f;
            return weight;
        }

        private static float Sigmoid(float x)
        {
            var exp = MathF.Exp(-x);
            return 2f / (1f + exp) - 1f;
        }

        private static float DerivativeSigmoid(float x)
        {
            return 0.5f * (1f + x) * (1f - x);
        }
    }

    public class Layer
    {
        public Neuron[] Neurons { get; set; }

        [JsonIgnore]
        public float[] Output { get; set; }

        public static Layer CreateRandom(int inputCount, int neuronCount, Random rng)
        {
            var neurons = new Neuron[neuronCount];
            var layer = new Layer
            {
                Neurons = neurons,
                Output = new float[neuronCount],
            };

            for (int i = 0; i < neuronCount; i++)
            {
                neurons[i] = Neuron.CreateRandom(inputCount, rng);
            }

            return layer;
        }
    }

    public class TrainedLayer
    {
        public Neuron[] Deltas { get; set; }

        public float[] Gradients { get; set; }

        public float[] Derivatives { get; set; }

        public static TrainedLayer Create(int inputCount, int neuronCount)
        {
            var neurons = new Neuron[neuronCount];
            var layer = new TrainedLayer
            {
                Deltas = neurons,
                Gradients = new float[neuronCount],
                Derivatives = new float[neuronCount],
            };

            for (int i = 0; i < neuronCount; i++)
            {
                neurons[i] = Neuron.Empty(inputCount);
            }

            return layer;
        }
    }

    public class Neuron
    {
        public float[] Weights { get; set; }

        public float Bias { get; set; }

        public static Neuron CreateRandom(int inputCount, Random rng)
        {
            var weights = new float[inputCount];
            var neuron = new Neuron
            {
                Bias = GetRandomFloat(-1, 1, rng),
                Weights = weights,
            };

            for (int i = 0; i < inputCount; i++)
            {
                weights[i] = GetRandomFloat(-1, 1, rng);
            }

            return neuron;
        }

        public static Neuron Empty(int inputCount)
        {
            var neuron = new Neuron
            {
                Weights = new float[inputCount],
            };

            return neuron;
        }

        private static float GetRandomFloat(float min, float max, Random rng)
        {
            return (float)(rng.NextDouble() * (max - min) + min);
        }
    }
}
