using System;
using System.Collections.Generic;
using System.Linq;

namespace Visionary.ConvolutionalNeuralNetwork
{
	using System.Threading;

    public class Layer
	{
		public void PeriodicWeightSanityCheck()
		{
			foreach (var weight in m_Weights)
			{
				double val = Math.Abs(weight.value);

				if ((val > 100.0) && (m_bFloatingPointWarning == false))
				{
					// 100.0 is an arbitrary value, that no reasonable weight should ever exceed

					//string strMess;
					//strMess.Format( _T( "Caution: Weights are becoming unboundedly large \n" )
					//    _T( "Layer: %s \nWeight: %s \nWeight value = %g \nWeight Hessian = %g\n\n" )
					//    _T( "Suggest abandoning this backpropagation and investigating" ),
					//    label.c_str(), ww.label.c_str(), ww.value, ww.diagHessian );

					//::MessageBox( NULL, strMess, _T( "Problem With Weights" ), MB_ICONEXCLAMATION | MB_OK );

					m_bFloatingPointWarning = true;
				}
			}
		}

		public void Calculate()
		{
			foreach (var neuron in m_Neurons)
			{
				double dSum = m_Weights[(int)neuron.m_Connections[0].WeightIndex].value
					   + neuron.m_Connections
							.Skip(1)
							.Sum(connection => m_Weights[(int)connection.WeightIndex].value
													* m_pPrevLayer.m_Neurons[(int)connection.NeuronIndex].output);

				neuron.output = Utils.SIGMOID(dSum);
			}
		}

		public void Backpropagate(
			List<double> dErr_wrt_dXn /* in */,
			List<double> dErr_wrt_dXnm1 /* out */,
			List<double> thisLayerOutput,
			List<double> prevLayerOutput,
			double etaLearningRate)
		{
			int ii, jj;
			uint kk;
			int nIndex;
			double output;

			List<double> dErr_wrt_dYn = new List<double>(m_Neurons.Count);

			double[] dErr_wrt_dWn = new double[4 * m_Weights.Count]; // _alloca( sizeof(double) *  m_Weights.size() ) ];

			for (ii = 0; ii < m_Weights.Count; ++ii)
			{
				dErr_wrt_dWn[ii] = 0.0;
			}

			bool memorized = thisLayerOutput != null && prevLayerOutput != null;


			for (ii = 0; ii < m_Neurons.Count; ++ii)
			{
				output = memorized ? thisLayerOutput[ii] : this.m_Neurons[ii].output;

				dErr_wrt_dYn[ii] = Utils.DSIGMOID(output) * dErr_wrt_dXn[ii];
			}

			ii = 0;
			foreach (var neuron in m_Neurons)
			{
				foreach (var connection in neuron.m_Connections)
				{
					kk = connection.NeuronIndex;
					if (kk == UInt32.MaxValue)
					{
						output = 1.0;  // this is the bias weight
					}
					else
					{
						output = memorized != false ? prevLayerOutput[(int)kk] : this.m_pPrevLayer.m_Neurons[(int)kk].output;
					}

					dErr_wrt_dWn[connection.WeightIndex] += dErr_wrt_dYn[ii] * output;
				}

				ii++;
			}


			// calculate dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn, which is needed as the input value of
			// dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
			// For each neuron in this layer

			foreach (var neuron in m_Neurons)
			{
				foreach (var connection in neuron.m_Connections)
				{
					kk = connection.NeuronIndex;
					if (kk != UInt32.MaxValue)
					{
						nIndex = (int)kk;
						dErr_wrt_dXnm1[nIndex] += dErr_wrt_dYn[ii] * m_Weights[(int)connection.WeightIndex].value;
					}
				}
			}

			double dMicron = 0.1; //TODO ::GetPreferences().m_dMicronLimitParameter;

			for (jj = 0; jj < m_Weights.Count; ++jj)
			{
				double divisor = this.m_Weights[jj].diagHessian + dMicron;

				double epsilon = etaLearningRate / divisor;
				double oldValue = this.m_Weights[jj].value;
				double newValue = oldValue - epsilon * dErr_wrt_dWn[jj];

				while (oldValue != Interlocked.CompareExchange(ref m_Weights[jj].value, newValue, oldValue))
				{
					// another thread must have modified the weight.  Obtain its new value, adjust it, and try again
					oldValue = m_Weights[jj].value;
					newValue = oldValue - epsilon * dErr_wrt_dWn[jj];
				}
			}
		}

		public void EraseHessianInformation()
		{
			foreach (var weight in m_Weights)
			{
				weight.diagHessian = 0.0;
			}
		}

		public void DivideHessianInformationBy(double divisor)
		{


		}

		public void BackpropagateSecondDerivatives(
			List<double> d2Err_wrt_dXn /* in */,
			List<double> d2Err_wrt_dXnm1 /* out */)
		{
			int ii, jj;
			uint kk;
			int nIndex;
			double output;
			double dTemp;

			List<double> d2Err_wrt_dYn = new List<double>(m_Neurons.Count);

			double[] d2Err_wrt_dWn = new double[sizeof(double) * m_Weights.Count];

			for (ii = 0; ii < m_Weights.Count; ++ii)
			{
				d2Err_wrt_dWn[ii] = 0.0;
			}

			for (ii = 0; ii < m_Neurons.Count; ++ii)
			{
				output = m_Neurons[ii].output;

				dTemp = Utils.DSIGMOID(output);
				d2Err_wrt_dYn[ii] = d2Err_wrt_dXn[ii] * dTemp * dTemp;
			}

			ii = 0;
			foreach (var neuron in m_Neurons)
			{
				foreach (var connection in neuron.m_Connections)
				{
					kk = connection.NeuronIndex;
					output = kk == UInt32.MaxValue ? 1.0 : this.m_pPrevLayer.m_Neurons[(int)kk].output;
					d2Err_wrt_dWn[connection.WeightIndex] += d2Err_wrt_dYn[ii] * output * output;
				}

				ii++;
			}

			ii = 0;
			foreach (var neuron in m_Neurons)
			{
				foreach (var connection in neuron.m_Connections)
				{
					kk = connection.NeuronIndex;
					if (kk != UInt32.MaxValue)
					{
						nIndex = (int)kk;

						dTemp = m_Weights[(int)connection.WeightIndex].value;
						d2Err_wrt_dXnm1[nIndex] += d2Err_wrt_dYn[ii] * dTemp * dTemp;
					}

				}

				ii++;
			}

			for (jj = 0; jj < m_Weights.Count; ++jj)
			{
				double oldValue = this.m_Weights[jj].diagHessian;
				double newValue = oldValue + d2Err_wrt_dWn[jj];

				while (oldValue != Interlocked.CompareExchange(ref m_Weights[jj].diagHessian, newValue, oldValue))
				{
					oldValue = m_Weights[jj].diagHessian;
					newValue = oldValue + d2Err_wrt_dWn[jj];
				}
			}
		}

		// void Serialize(CArchive& ar );

		public Layer()
		{

		}

		public Layer(string str, Layer pPrev = null)
		{
			label = str;
			m_pPrevLayer = pPrev;
		}

		public List<Weight> m_Weights = new List<Weight>();

		public List<Neuron> m_Neurons = new List<Neuron>();

		public string label = String.Empty;

		public Layer m_pPrevLayer = null;

		public bool m_bFloatingPointWarning; // flag for one-time warning (per layer) about potential floating point overflow

		public void Initialize()
		{
			m_Weights.Clear();
			m_Neurons.Clear();
			m_bFloatingPointWarning = false;
		}
	}
}