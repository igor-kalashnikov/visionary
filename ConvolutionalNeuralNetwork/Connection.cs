using System;

namespace Visionary.ConvolutionalNeuralNetwork
{
    /// <summary>
    /// Class represents connection to neuron in neural network.
    /// </summary>
    public class Connection
    {
        public Connection(uint neuron = UInt32.MaxValue, uint weight = UInt32.MaxValue)
        {
            NeuronIndex = neuron;
            WeightIndex = weight;
        }

        public uint WeightIndex { get; set; }

        public uint NeuronIndex { get; set; }
    }
}
