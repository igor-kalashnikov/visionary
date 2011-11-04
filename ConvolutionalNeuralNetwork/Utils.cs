// --------------------------------------------------------------------------------------------------------------------
// <copyright file="Utils.cs" company="">
//   
// </copyright>
// <summary>
//   The utility class.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace Visionary.ConvolutionalNeuralNetwork
{
    using System;

    /// <summary>
    /// The utility class.
    /// </summary>
    public static class Utils
    {
        #region Constants and Fields

        /// <summary>
        /// The random object.
        /// </summary>
        private static readonly Random random = new Random();

        #endregion

        #region Public Methods

        /// <summary>
        /// The random double between min and max.
        /// </summary>
        /// <param name="min">
        /// The minimum value.
        /// </param>
        /// <param name="max">
        /// The maximum value.
        /// </param>
        /// <returns>
        /// The random double.
        /// </returns>
        public static double RandomDouble(double min, double max)
        {
            return random.NextDouble() * (max - min) + min;
        }

        /// <summary>
        /// The sigmoid function.
        /// </summary>
        /// <param name="x">
        /// The argument.
        /// </param>
        /// <returns>
        /// The sigmoid.
        /// </returns>
        public static double Sigmoid(double x)
        {
            return 1.7159 * Math.Tanh(0.66666667 * x);
        }

        /// <summary>
        /// The sigmoid function derivative.
        /// </summary>
        /// <param name="x">
        /// The argument.
        /// </param>
        /// <returns>
        /// The sigmoid derivative.
        /// </returns>
        public static double SigmoidDerivative(double x)
        {
            return 0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x);
        }

        public static double UNIFORM_PLUS_MINUS_ONE()
        {
            return random.NextDouble() * 2 - 1;
        }

        public static double UNIFORM_ZERO_THRU_ONE()
        {
            return random.NextDouble();
        }

        #endregion
    }
}