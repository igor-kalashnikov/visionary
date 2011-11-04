// --------------------------------------------------------------------------------------------------------------------
// <copyright file="Weight.cs" company="">
//   
// </copyright>
// <summary>
//   Class represents weight in neural network. May be shares for several connections.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace Visionary.ConvolutionalNeuralNetwork
{
    using System;

    /// <summary>
    /// Class represents weight in neural network. May be shares for several connections.
    /// </summary>
    public class Weight
    {
        #region Constants and Fields

        /// <summary>
        /// The diag hessian.
        /// </summary>
        public double diagHessian;

        /// <summary>
        /// The weight value.
        /// </summary>
        public double value;

        /// <summary>
        /// The label.
        /// </summary>
        private string label = string.Empty;

        #endregion

        #region Constructors and Destructors

        /// <summary>
        /// Initializes a new instance of the <see cref="Weight"/> class with null value and empty label.
        /// </summary>
        public Weight()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Weight"/> class.
        /// </summary>
        /// <param name="label">
        /// The label.
        /// </param>
        /// <param name="value">
        /// The value.
        /// </param>
        public Weight(string label, double value = 0.0)
        {
            this.Label = label;
            this.value = value;
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets or sets Label.
        /// </summary>
        public string Label
        {
            get
            {
                return this.label;
            }

            set
            {
                this.label = value;
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// The initialize.
        /// </summary>
        private void Initialize()
        {
        }

        #endregion
    }
}