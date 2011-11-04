// --------------------------------------------------------------------------------------------------------------------
// <copyright file="Layer.cs" company="">
//   
// </copyright>
// <summary>
//   Class represents layer in neural network. Contains neurons and weights.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace Visionary.ConvolutionalNeuralNetwork
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;

    /// <summary>
    /// Class represents layer in neural network. Contains neurons and weights.
    /// </summary>
    public class Layer
    {
        #region Constants and Fields

        /// <summary>
        /// The m_b floating point warning.
        /// </summary>
        public bool m_bFloatingPointWarning;
                    // flag for one-time warning (per layer) about potential floating point overflow

        /// <summary>
        /// The label.
        /// </summary>
        private string label = string.Empty;

        /// <summary>
        /// The m_ neurons.
        /// </summary>
        private List<Neuron> neurons = new List<Neuron>();

        /// <summary>
        /// The m_ weights.
        /// </summary>
        private List<Weight> weights = new List<Weight>();

        /// <summary>
        /// The previous layer of neural network.
        /// </summary>
        private Layer previousLayer;

        #endregion

        #region Constructors and Destructors

        /// <summary>
        /// Initializes a new instance of the <see cref="Layer"/> class.
        /// </summary>
        public Layer()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Layer"/> class.
        /// </summary>
        /// <param name="str">
        /// The str.
        /// </param>
        /// <param name="pPrev">
        /// The p prev.
        /// </param>
        public Layer(string str, Layer pPrev = null)
        {
            this.label = str;
            this.previousLayer = pPrev;
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

        /// <summary>
        /// Gets or sets Neurons.
        /// </summary>
        public List<Neuron> Neurons
        {
            get
            {
                return this.neurons;
            }

            set
            {
                this.neurons = value;
            }
        }

        /// <summary>
        /// Gets or sets PreviousLayer.
        /// </summary>
        public Layer PreviousLayer
        {
            get
            {
                return this.previousLayer;
            }

            set
            {
                this.previousLayer = value;
            }
        }

        /// <summary>
        /// Gets or sets Weights.
        /// </summary>
        public List<Weight> Weights
        {
            get
            {
                return this.weights;
            }

            set
            {
                this.weights = value;
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// The backpropagate.
        /// </summary>
        /// <param name="dErr_wrt_dXn">
        /// The d err_wrt_d xn.
        /// </param>
        /// <param name="dErr_wrt_dXnm1">
        /// The d err_wrt_d xnm 1.
        /// </param>
        /// <param name="thisLayerOutput">
        /// This layer output.
        /// </param>
        /// <param name="prevLayerOutput">
        /// The previous layer output.
        /// </param>
        /// <param name="etaLearningRate">
        /// The eta learning rate.
        /// </param>
        public void Backpropagate(
            double[] dErr_wrt_dXn /* in */,
            double[] dErr_wrt_dXnm1 /* out */,
            List<double> thisLayerOutput,
            List<double> prevLayerOutput, 
            double etaLearningRate)
        {
            int ii, jj;
            uint kk;
            double output;

            var dErr_wrt_dYn = new double[this.neurons.Count];

            var dErr_wrt_dWn = new double[sizeof(double) * this.weights.Count];
                
                // _alloca( sizeof(double) *  weights.size() ) ];
            for (ii = 0; ii < this.weights.Count; ++ii)
            {
                dErr_wrt_dWn[ii] = 0.0;
            }

            bool memorized = thisLayerOutput != null && prevLayerOutput != null;

            for (ii = 0; ii < this.neurons.Count; ++ii)
            {
                output = memorized ? thisLayerOutput[ii] : this.neurons[ii].Output;
                dErr_wrt_dYn[ii] = Utils.SigmoidDerivative(output) * dErr_wrt_dXn[ii];
            }

            ii = 0;
            foreach (Neuron neuron in this.neurons)
            {
                foreach (Connection connection in neuron.Connections)
                {
                    kk = connection.NeuronIndex;
                    if (kk == uint.MaxValue)
                    {
                        output = 1.0; // this is the bias weight
                    }
                    else
                    {
                        output = memorized ? prevLayerOutput[(int)kk] : this.previousLayer.neurons[(int)kk].Output;
                    }

                    dErr_wrt_dWn[connection.WeightIndex] += dErr_wrt_dYn[ii] * output;
                }

                ii++;
            }

            // calculate dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn, which is needed as the input value of
            // dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
            // For each neuron in this layer
            ii = 0;
            foreach (Neuron neuron in this.neurons)
            {
                foreach (Connection connection in neuron.Connections)
                {
                    kk = connection.NeuronIndex;
                    if (kk != uint.MaxValue)
                    {
                        var nIndex = (int)kk;
                        dErr_wrt_dXnm1[nIndex] += dErr_wrt_dYn[ii] * this.weights[(int)connection.WeightIndex].value;
                    }
                }
            }

            double dMicron = 0.1; // TODO ::GetPreferences().m_dMicronLimitParameter;

            for (jj = 0; jj < this.weights.Count; ++jj)
            {
                double divisor = this.weights[jj].diagHessian + dMicron;
                double epsilon = etaLearningRate / divisor;
                double oldValue = this.weights[jj].value;
                double newValue = oldValue - epsilon * dErr_wrt_dWn[jj];

                while (oldValue != Interlocked.CompareExchange(ref this.weights[jj].value, newValue, oldValue))
                {
                    // another thread must have modified the weight.  Obtain its new value, adjust it, and try again
                    oldValue = this.weights[jj].value;
                    newValue = oldValue - epsilon * dErr_wrt_dWn[jj];
                }
            }
        }

        /// <summary>
        /// The backpropagate second derivatives.
        /// </summary>
        /// <param name="d2Err_wrt_dXn">
        /// The d 2 err_wrt_d xn.
        /// </param>
        /// <param name="d2Err_wrt_dXnm1">
        /// The d 2 err_wrt_d xnm 1.
        /// </param>
        public void BackpropagateSecondDerivatives(
            List<double> d2Err_wrt_dXn /* in */, List<double> d2Err_wrt_dXnm1 /* out */)
        {
            int ii, jj;
            uint kk;
            int nIndex;
            double output;
            double dTemp;

            var d2Err_wrt_dYn = new List<double>(this.neurons.Count);

            var d2Err_wrt_dWn = new double[sizeof(double) * this.weights.Count];

            for (ii = 0; ii < this.weights.Count; ++ii)
            {
                d2Err_wrt_dWn[ii] = 0.0;
            }

            for (ii = 0; ii < this.neurons.Count; ++ii)
            {
                output = this.neurons[ii].Output;

                dTemp = Utils.SigmoidDerivative(output);
                d2Err_wrt_dYn[ii] = d2Err_wrt_dXn[ii] * dTemp * dTemp;
            }

            ii = 0;
            foreach (Neuron neuron in this.neurons)
            {
                foreach (Connection connection in neuron.Connections)
                {
                    kk = connection.NeuronIndex;
                    output = kk == uint.MaxValue ? 1.0 : this.previousLayer.neurons[(int)kk].Output;
                    d2Err_wrt_dWn[connection.WeightIndex] += d2Err_wrt_dYn[ii] * output * output;
                }

                ii++;
            }

            ii = 0;
            foreach (Neuron neuron in this.neurons)
            {
                foreach (Connection connection in neuron.Connections)
                {
                    kk = connection.NeuronIndex;
                    if (kk != uint.MaxValue)
                    {
                        nIndex = (int)kk;

                        dTemp = this.weights[(int)connection.WeightIndex].value;
                        d2Err_wrt_dXnm1[nIndex] += d2Err_wrt_dYn[ii] * dTemp * dTemp;
                    }
                }

                ii++;
            }

            for (jj = 0; jj < this.weights.Count; ++jj)
            {
                double oldValue = this.weights[jj].diagHessian;
                double newValue = oldValue + d2Err_wrt_dWn[jj];

                while (oldValue != Interlocked.CompareExchange(ref this.weights[jj].diagHessian, newValue, oldValue))
                {
                    oldValue = this.weights[jj].diagHessian;
                    newValue = oldValue + d2Err_wrt_dWn[jj];
                }
            }
        }

        /// <summary>
        /// The calculate.
        /// </summary>
        public void Calculate()
        {
            foreach (Neuron neuron in this.neurons)
            {
                double dSum = this.weights[(int)neuron.Connections[0].WeightIndex].value
                              +
                              neuron.Connections.Skip(1).Sum(
                                  connection =>
                                  this.weights[(int)connection.WeightIndex].value
                                  * this.previousLayer.neurons[(int)connection.NeuronIndex].Output);

                neuron.Output = Utils.Sigmoid(dSum);
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
        }

        /// <summary>
        /// The erase hessian information.
        /// </summary>
        public void EraseHessianInformation()
        {
            foreach (Weight weight in this.weights)
            {
                weight.diagHessian = 0.0;
            }
        }

        /// <summary>
        /// The initialize.
        /// </summary>
        public void Initialize()
        {
            this.weights.Clear();
            this.neurons.Clear();
            this.m_bFloatingPointWarning = false;
        }

        /// <summary>
        /// The periodic weight sanity check.
        /// </summary>
        public void PeriodicWeightSanityCheck()
        {
            foreach (Weight weight in this.weights)
            {
                double val = Math.Abs(weight.value);

                if ((val > 100.0) && (this.m_bFloatingPointWarning == false))
                {
                    // 100.0 is an arbitrary value, that no reasonable weight should ever exceed

                    // string strMess;
                    // strMess.Format( _T( "Caution: Weights are becoming unboundedly large \n" )
                    // _T( "Layer: %s \nWeight: %s \nWeight value = %g \nWeight Hessian = %g\n\n" )
                    // _T( "Suggest abandoning this backpropagation and investigating" ),
                    // label.c_str(), ww.label.c_str(), ww.value, ww.diagHessian );

                    // ::MessageBox( NULL, strMess, _T( "Problem With Weights" ), MB_ICONEXCLAMATION | MB_OK );
                    this.m_bFloatingPointWarning = true;
                }
            }
        }

        #endregion
    }
}