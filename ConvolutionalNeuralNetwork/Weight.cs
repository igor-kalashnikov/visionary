using System;

namespace Visionary.ConvolutionalNeuralNetwork
{
    public class Weight
    {
        public Weight()
        {

        }

        public Weight(string str, double val = 0.0)
        {
            label = str;
            value = val;
        }

        string label = String.Empty;
        public double value = 0.0;
        public double diagHessian = 0.0;


        void Initialize()
        {

        }
    }
}
