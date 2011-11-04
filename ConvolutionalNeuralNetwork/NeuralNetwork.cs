// --------------------------------------------------------------------------------------------------------------------
// <copyright file="NeuralNetwork.cs" company="">
//   
// </copyright>
// <summary>
//   The neural network.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace Visionary.ConvolutionalNeuralNetwork
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// The neural network.
    /// </summary>
    public class NeuralNetwork
    {
        #region Constants and Fields

        /// <summary>
        /// The backpropagations count.
        /// </summary>
        private uint backpropagations; // counter used in connection with Weight sanity check

        /// <summary>
        /// The eta learning rate.
        /// </summary>
        private double etaLearningRate = 0.3;

        /// <summary>
        /// The layers.
        /// </summary>
        private List<Layer> layers = new List<Layer>();

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets or sets EtaLearningRate.
        /// </summary>
        public double EtaLearningRate
        {
            get
            {
                return this.etaLearningRate;
            }

            set
            {
                this.etaLearningRate = value;
            }
        }

        /// <summary>
        /// Gets or sets EtaLearningRatePrevious.
        /// </summary>
        public double EtaLearningRatePrevious { get; set; }

        /// <summary>
        /// Gets or sets Layers.
        /// </summary>
        public List<Layer> Layers
        {
            get
            {
                return this.layers;
            }

            set
            {
                this.layers = value;
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// The backpropagate.
        /// </summary>
        /// <param name="actualOutput">
        /// The actual output.
        /// </param>
        /// <param name="desiredOutput">
        /// The desired output.
        /// </param>
        /// <param name="count">
        /// The count.
        /// </param>
        /// <param name="pMemorizedNeuronOutputs">
        /// The p memorized neuron outputs.
        /// </param>
        public void Backpropagate(
            double[] actualOutput, double[] desiredOutput, uint count, List<List<double>> pMemorizedNeuronOutputs = null)
        {
            if (actualOutput == null || desiredOutput == null || count >= 256)
            {
                return;
            }

            this.backpropagations++;

            if ((this.backpropagations % 10000) == 0)
            {
                this.PeriodicWeightSanityCheck();
            }

            var dErr_wrt_dXlast = new double[this.layers[this.layers.Count - 1].Neurons.Count];
            var differentials = new double[this.layers.Count][];

            for (int i = 0; i < this.layers[this.layers.Count - 1].Neurons.Count; ++i)
            {
                dErr_wrt_dXlast[i] = actualOutput[i] - desiredOutput[i];
            }

            differentials[layers.Count - 1] = dErr_wrt_dXlast; // last one

            for (int i = 0; i < layers.Count - 1; ++i)
            {
                differentials[i] = new double[this.layers[i].Neurons.Count];
            }

            bool bMemorized = pMemorizedNeuronOutputs != null;
            for (int i = layers.Count - 1; i > 0; i--)
            {
                if (bMemorized)
                {
                    this.layers[i].Backpropagate(
                        differentials[i],
                        differentials[i - 1],
                        pMemorizedNeuronOutputs[i],
                        pMemorizedNeuronOutputs[i - 1],
                        this.etaLearningRate);
                }
                else
                {
                    this.layers[i].Backpropagate(
                        differentials[i],
                        differentials[i - 1],
                        null,
                        null,
                        this.etaLearningRate);
                }
            }
        }

        /// <summary>
        /// The backpropagate second dervatives.
        /// </summary>
        /// <param name="actualOutputVector">
        /// The actual output vector.
        /// </param>
        /// <param name="targetOutputVector">
        /// The target output vector.
        /// </param>
        /// <param name="count">
        /// The count.
        /// </param>
        public void BackpropagateSecondDervatives(double[] actualOutputVector, double[] targetOutputVector, uint count)
        {
            if (actualOutputVector == null || targetOutputVector == null || count >= 256)
            {
                return;
            }

            var d2Err_wrt_dXlast = new List<double>(this.layers[this.layers.Count - 1].Neurons.Count);
            var differentials = new List<List<double>>(this.layers.Count);

            for (int i = 0; i < this.layers[this.layers.Count - 1].Neurons.Count; ++i)
            {
                d2Err_wrt_dXlast[i] = 1.0;
            }

            differentials[layers.Count - 1] = d2Err_wrt_dXlast; // last one

            for (int i = 0; i < layers.Count - 1; ++i)
            {
                differentials[i].Capacity = this.layers[i].Neurons.Count;
            }

            for (int i = layers.Count - 1; i > 0; i--)
            {
                this.layers[i].BackpropagateSecondDerivatives(differentials[i], differentials[i - 1]);
            }

            differentials.Clear();
        }

        /// <summary>
        /// The calculate.
        /// </summary>
        /// <param name="inputVector">
        /// The input vector.
        /// </param>
        /// <param name="iCount">
        /// The i count.
        /// </param>
        /// <param name="outputVector">
        /// The output vector.
        /// </param>
        /// <param name="oCount">
        /// The o count.
        /// </param>
        /// <param name="pNeuronOutputs">
        /// The p neuron outputs.
        /// </param>
        public void Calculate(
            double[] inputVector,
            uint iCount,
            double[] outputVector = null,
            uint oCount = 0,
            List<List<double>> pNeuronOutputs = null)
        {
            int count = 0;
            foreach (Neuron neuron in this.layers[0].Neurons)
            {
                if (count < iCount)
                {
                    neuron.Output = inputVector[count];
                }

                count++;
            }

            foreach (Layer layer in this.layers.Where(layer => !layer.Equals(this.layers[0])))
            {
                layer.Calculate();
            }

            // load up output vector with results
            if (outputVector != null)
            {
                int ii = 0;
                foreach (Neuron neuron in this.layers[this.layers.Count - 1].Neurons)
                {
                    if (ii < oCount)
                    {
                        outputVector[ii] = neuron.Output;
                    }

                    ii++;
                }
            }

            // load up neuron output values with results
            if (pNeuronOutputs != null)
            {
                // check for first time use (re-use is expected)
                if (pNeuronOutputs.Count == 0)
                {
                    // it's empty, so allocate memory for its use
                    pNeuronOutputs.Clear(); // for safekeeping
                    pNeuronOutputs.AddRange(
                        this.layers.Select(layer => layer.Neurons.Select(neuron => neuron.Output).ToList()));
                }
                else
                {
                    // it's not empty, so assume it's been used in a past iteration and memory for
                    // it has already been allocated internally.  Simply store the values
                    int ii = 0, jj = 0;
                    foreach (Layer layer in this.layers)
                    {
                        foreach (Neuron neuron in layer.Neurons)
                        {
                            pNeuronOutputs[jj][ii] = neuron.Output;
                            ++ii;
                        }

                        ++jj;
                    }
                }
            }
        }

        /// <summary>
        /// The divide hessian information by.
        /// </summary>
        /// <param name="divisor">
        /// The divisor.
        /// </param>
        public void DivideHessianInformationBy(double divisor)
        {
            foreach (Layer layer in this.layers)
            {
                layer.DivideHessianInformationBy(divisor);
            }
        }

        /// <summary>
        /// The erase hessian information.
        /// </summary>
        public void EraseHessianInformation()
        {
            // controls each layer to erase (set to value of zero) all its diagonal Hessian info
            foreach (Layer layer in this.layers)
            {
                layer.EraseHessianInformation();
            }
        }

        // void Serialize(CArchive &ar);

        /// <summary>
        /// The initialize.
        /// </summary>
        public void Initialize()
        {
            this.layers.Clear();

            //this.etaLearningRate = .001;

            // arbitrary, so that brand-new NNs can be serialized with a non-ridiculous number
            this.backpropagations = 0;
        }

        #endregion

        #region Methods

        /// <summary>
        /// The periodic weight sanity check.
        /// </summary>
        private void PeriodicWeightSanityCheck()
        {
            foreach (Layer layer in this.layers)
            {
                layer.PeriodicWeightSanityCheck();
            }
        }

        #endregion
    }
}