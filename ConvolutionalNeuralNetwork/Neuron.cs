// --------------------------------------------------------------------------------------------------------------------
// <copyright file="Neuron.cs" company="">
//   
// </copyright>
// <summary>
//   The neuron.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace Visionary.ConvolutionalNeuralNetwork
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// The neuron.
    /// </summary>
    public class Neuron
    {
        #region Constants and Fields

        /// <summary>
        /// The connections.
        /// </summary>
        private List<Connection> connections = new List<Connection>();

        /// <summary>
        /// The label.
        /// </summary>
        private string label = string.Empty;

        #endregion

        #region Constructors and Destructors

        /// <summary>
        /// Initializes a new instance of the <see cref="Neuron"/> class.
        /// </summary>
        public Neuron()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Neuron"/> class.
        /// </summary>
        /// <param name="str">
        /// The str.
        /// </param>
        public Neuron(string str)
        {
            this.label = str;
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets or sets Connections.
        /// </summary>
        public List<Connection> Connections
        {
            get
            {
                return this.connections;
            }

            set
            {
                this.connections = value;
            }
        }

        /// <summary>
        /// Gets or sets Output.
        /// </summary>
        public double Output { get; set; }

        #endregion

        #region Public Methods

        /// <summary>
        /// The add connection.
        /// </summary>
        /// <param name="neuronIndex">
        /// The neuron index.
        /// </param>
        /// <param name="weightIndex">
        /// The weight index.
        /// </param>
        public void AddConnection(uint neuronIndex, uint weightIndex)
        {
            this.connections.Add(new Connection(neuronIndex, weightIndex));
        }

        /// <summary>
        /// The add connection.
        /// </summary>
        /// <param name="connection">
        /// The connection.
        /// </param>
        public void AddConnection(Connection connection)
        {
            this.connections.Add(connection);
        }

        #endregion

        #region Methods

        /// <summary>
        /// The initialize.
        /// </summary>
        private void Initialize()
        {
            this.connections.Clear();
        }

        #endregion
    }
}