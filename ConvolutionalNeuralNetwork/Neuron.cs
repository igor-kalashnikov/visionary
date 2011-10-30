using System;
using System.Collections.Generic;

namespace Visionary.ConvolutionalNeuralNetwork
{
    public class Neuron
    {
        public Neuron()
        {

        }

        public Neuron(string str)
        {
            label = str;
        }

        public  void AddConnection(uint iNeuron, uint iWeight)
        {
            m_Connections.Add(new Connection(iNeuron, iWeight));
        }

        public void AddConnection(Connection conn)
        {
            m_Connections.Add(conn);
        }


        string label = String.Empty;
        public double output = 0.0;

        public List<Connection> m_Connections = new List<Connection>();

        ///	VectorWeights m_Weights;
        ///	VectorNeurons m_Neurons;

        void Initialize()
        {
            m_Connections.Clear();
        }
    }
}
