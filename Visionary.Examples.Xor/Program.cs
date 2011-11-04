using Visionary.ConvolutionalNeuralNetwork;
namespace Visionary.Examples.Xor
{
    using System;

    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork();
            Layer layer = new Layer();
            for (int i = 0; i < 2; ++i)
            {
                layer.Neurons.Add(new Neuron());
            }

            neuralNetwork.Layers.Add(layer);

            //--------------------------------------------------------------------------------------------
            layer = new Layer("", layer);

            for (int i = 0; i < 2; ++i)
            {
                layer.Neurons.Add(new Neuron());
            }

            for (int i = 0; i < layer.Neurons.Count * (layer.PreviousLayer.Neurons.Count + 1); ++i)
            {
                layer.Weights.Add(new Weight(string.Empty, Utils.RandomDouble(-0.3, 0.3)));
            }

            for (int i = 0; i < layer.Neurons.Count; ++i)
            {
                layer.Neurons[i].AddConnection(UInt32.MaxValue, (uint)(i * layer.Neurons.Count)); // bias weight
                for (int j = 0; j < layer.PreviousLayer.Neurons.Count; ++j)
                {
                    layer.Neurons[i].AddConnection(new Connection((uint)i, (uint)(i * layer.Neurons.Count + j + 1)));
                }
            }

            neuralNetwork.Layers.Add(layer);

            layer = new Layer(string.Empty, layer);

            for (int i = 0; i < 1; ++i)
            {
                layer.Neurons.Add(new Neuron());
            }

            for (int i = 0; i < layer.Neurons.Count * (layer.PreviousLayer.Neurons.Count + 1); ++i)
            {
                layer.Weights.Add(new Weight(string.Empty, Utils.RandomDouble(-0.3, 0.3)));
            }

            for (int i = 0; i < layer.Neurons.Count; ++i)
            {
                layer.Neurons[i].AddConnection(UInt32.MaxValue, (uint)(i * layer.Neurons.Count)); // bias weight
                for (int j = 0; j < layer.PreviousLayer.Neurons.Count; ++j)
                {
                    layer.Neurons[i].AddConnection(new Connection((uint)i, (uint)(i * layer.Neurons.Count + j + 1)));
                }
            }

            neuralNetwork.Layers.Add(layer);

            double[] input = new double[]
                {
                    1, 1
                };

            double[] targetOutput = new double[]
                {
                    0
                };

            double[] output = new double[1];
            for (int i = 0; i < 100000; ++i)
            {
                neuralNetwork.Calculate(input, 100, output, 100);
                neuralNetwork.Backpropagate(output, targetOutput, 100);

                input = new double[] { 0, 1 };
                targetOutput = new double[] { 1 };

                neuralNetwork.Calculate(input, 100, output, 100);
                neuralNetwork.Backpropagate(output, targetOutput, 100);

                input = new double[] { 1, 0 };
                targetOutput = new double[] { 1 };

                neuralNetwork.Calculate(input, 100, output, 100);
                neuralNetwork.Backpropagate(output, targetOutput, 100);

                input = new double[] { 0, 0 };
                targetOutput = new double[] { 0 };

                neuralNetwork.Calculate(input, 100, output, 100);
                neuralNetwork.Backpropagate(output, targetOutput, 100);

                input = new double[] { 1, 1 };
                targetOutput = new double[] { 0 };
            }

            foreach (var l in neuralNetwork.Layers)
            {
                int i = 0;
                foreach (var weight in l.Weights)
                {
                    if (l.PreviousLayer != null && i > l.PreviousLayer.Neurons.Count)
                    {
                        Console.WriteLine();
                        i = 0;
                    }

                    Console.Write(weight.value + " ");
                    i++;
                }

                Console.WriteLine();
                Console.WriteLine();
            }

            input = new double[] { 1, 1 };
            neuralNetwork.Calculate(input, 100, output, 100);

            Console.WriteLine(output[0]);
        }
    }
}
