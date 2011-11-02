using System;

namespace Visionary.ConvolutionalNeuralNetwork
{
    public static class Utils
    {
        public static double SIGMOID(double x)
        {
            return 1.7159 * Math.Tanh(0.66666667 * x);
        }

        public static double DSIGMOID(double x)
        {
            return 0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x);
        }

        public static double UNIFORM_PLUS_MINUS_ONE()
        {
            Random random = new Random();
            return random.NextDouble() * 2 - 1;
        }

        public static double UNIFORM_ZERO_THRU_ONE()
        {
            Random random = new Random();
            return random.NextDouble();
        }
    }
}
