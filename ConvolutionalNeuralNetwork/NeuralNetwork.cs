namespace Visionary.ConvolutionalNeuralNetwork
{
    using System.Collections.Generic;
    using System.Linq;

    public class NeuralNetwork
    {
        public double m_etaLearningRatePrevious;
        public double m_etaLearningRate;

        private List<Layer> m_Layers = new List<Layer>();

        public List<Layer> Layers
        {
            get
            {
                return m_Layers;
            }

            set
            {
                m_Layers = value;
            }
        }

        uint m_cBackprops;  // counter used in connection with Weight sanity check

        void PeriodicWeightSanityCheck()
        {
            foreach (var layer in m_Layers)
            {
                layer.PeriodicWeightSanityCheck();
            }
        }

        public void Calculate(double[] inputVector, uint iCount,
            double[] outputVector = null, uint oCount = 0,
            List<List<double>> pNeuronOutputs = null)
        {
            int count = 0;
            foreach (var neuron in m_Layers[0].m_Neurons)
            {
                if (count < iCount)
                {
                    neuron.output = inputVector[count];
                }

                count++;
            }

            foreach (var layer in this.m_Layers.Where(layer => !layer.Equals(this.m_Layers[0])))
            {
                layer.Calculate();
            }

            // load up output vector with results
            if (outputVector != null)
            {
                int ii = 0;
                foreach (var neuron in m_Layers[m_Layers.Count - 1].m_Neurons)
                {
                    if (ii < oCount)
                    {
                        outputVector[ii] = neuron.output;
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
                    pNeuronOutputs.Clear();  // for safekeeping
                    pNeuronOutputs.AddRange(this.m_Layers.Select(layer => layer.m_Neurons.Select(neuron => neuron.output).ToList()));
                }
                else
                {
                    // it's not empty, so assume it's been used in a past iteration and memory for
                    // it has already been allocated internally.  Simply store the values

                    int ii = 0, jj = 0;
                    foreach (var layer in m_Layers)
                    {
                        foreach (var neuron in layer.m_Neurons)
                        {
                            pNeuronOutputs[jj][ii] = neuron.output;
                            ++ii;
                        }
                        ++jj;
                    }
                }
            }
        }

        public void Backpropagate(double[] actualOutput, double[] desiredOutput, uint count,
            List<List<double>> pMemorizedNeuronOutputs)
        {
            if (actualOutput == null || desiredOutput == null || count >= 256)
            {
                return;
            }

            m_cBackprops++;

            if ((m_cBackprops % 10000) == 0)
            {
                PeriodicWeightSanityCheck();
            }

            List<double> dErr_wrt_dXlast = new List<double>(m_Layers[m_Layers.Count - 1].m_Neurons.Count());
            List<List<double>> differentials = new List<List<double>>(m_Layers.Count);

            int iSize = m_Layers.Count;
            int ii;

            for (ii = 0; ii < m_Layers[m_Layers.Count - 1].m_Neurons.Count; ++ii)
            {
                dErr_wrt_dXlast[ii] = actualOutput[ii] - desiredOutput[ii];
            }

            differentials[iSize - 1] = dErr_wrt_dXlast;  // last one

            for (ii = 0; ii < iSize - 1; ++ii)
            {
                differentials[ii].Capacity = m_Layers[ii].m_Neurons.Count();
            }

            bool bMemorized = (pMemorizedNeuronOutputs != null);
            ii = iSize - 1;
            for (int ind = m_Layers.Count - 1; ind > 0; ind--)
            {
                if (bMemorized)
                {
                    m_Layers[ind].Backpropagate(differentials[ii], differentials[ii - 1],
                        pMemorizedNeuronOutputs[ii], pMemorizedNeuronOutputs[ii - 1], m_etaLearningRate);
                }
                else
                {
                    m_Layers[ind].Backpropagate(differentials[ii], differentials[ii - 1],
                        null, null, m_etaLearningRate);
                }

                --ii;
            }

            differentials.Clear();
        }

        public void EraseHessianInformation()
        {
            // controls each layer to erase (set to value of zero) all its diagonal Hessian info
            foreach (var layer in m_Layers)
            {
                layer.EraseHessianInformation();
            }
        }

        public void DivideHessianInformationBy(double divisor)
        {
            foreach (var layer in m_Layers)
            {
                layer.DivideHessianInformationBy(divisor);
            }
        }

        public void BackpropagateSecondDervatives(double[] actualOutputVector, double[] targetOutputVector, uint count)
        {
            if (actualOutputVector == null || targetOutputVector == null || count >= 256)
            {
                return;
            }

            List<double> d2Err_wrt_dXlast = new List<double>(m_Layers[m_Layers.Count - 1].m_Neurons.Count);
            List<List<double>> differentials = new List<List<double>>(m_Layers.Count);

            int iSize = m_Layers.Count;

            int ii;

            for (ii = 0; ii < m_Layers[m_Layers.Count - 1].m_Neurons.Count; ++ii)
            {
                d2Err_wrt_dXlast[ii] = 1.0;
            }

            differentials[iSize - 1] = d2Err_wrt_dXlast;  // last one

            for (ii = 0; ii < iSize - 1; ++ii)
            {
                differentials[ii].Capacity = m_Layers[ii].m_Neurons.Count;
            }

            ii = iSize - 1;
            for (int ind = m_Layers.Count - 1; ind > 0; ind--)
            {
                m_Layers[ind].BackpropagateSecondDerivatives(differentials[ii], differentials[ii - 1]);

                --ii;
            }

            differentials.Clear();
        }

        // void Serialize(CArchive &ar);

        public NeuralNetwork()
        {

        }

        void Initialize()
        {
            m_Layers.Clear();

            m_etaLearningRate = .001;  // arbitrary, so that brand-new NNs can be serialized with a non-ridiculous number
            m_cBackprops = 0;
        }

    }
}
