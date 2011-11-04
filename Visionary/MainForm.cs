using System.Windows.Forms;
using Visionary.Properties;
using System.Threading;
using System.Collections.Generic;
using System;
using Visionary.ConvolutionalNeuralNetwork;

namespace Visionary
{
    public partial class MainForm : Form
    {
        private NeuralNetwork m_NN = new NeuralNetwork();

        private static int GAUSSIAN_FIELD_SIZE = 21;

        private volatile uint m_cBackprops;

        private volatile bool m_bNeedHessian;

        private uint m_nAfterEveryNBackprops;

        private double m_dEtaDecay;

        private double m_dMinimumEta;

        private double m_dEstimatedCurrentMSE; // this number will be changed by one thread and used by others

        private static uint m_iBackpropThreadIdentifier; // static member used by threads to identify themselves

        private uint m_iNumBackpropThreadsRunning;

        private Thread[] m_pBackpropThreads = new Thread[100];

        private bool m_bDistortTrainingPatterns;

        private bool m_bBackpropThreadsAreRunning;

        private volatile bool m_bBackpropThreadAbortFlag;

        private uint m_iNextTrainingPattern;

        private uint[] m_iRandomizedTrainingPatternSequence = new uint[60000];

        private double[] m_DispH; // horiz distortion map array

        private double[] m_DispV; // vert distortion map array

        private double[,] m_GaussianKernel = new double[GAUSSIAN_FIELD_SIZE, GAUSSIAN_FIELD_SIZE];

        private int m_cCols; // size of the distortion maps

        private int m_cRows;

        private int m_cCount;

        private object m_bFilesOpen;

        private bool m_bTestingThreadsAreRunning;

        private bool m_bTestingThreadAbortFlag;

        private int m_iNextTestingPattern;

        private int m_iNumTestingThreadsRunning;

        private int m_iTestingThreadIdentifier;

        private decimal m_iWhichImageSet;

        private bool m_bDistortTestingPatterns;

        private Thread[] m_pTestingThreads = new Thread[100];

        const uint g_cImageSize = 28;
        const uint g_cVectorSize = 29;


        public MainForm()
        {
            InitializeComponent();
            Init();
            InitNeuralNetwork();
        }

        private void Init()
        {
            // TODO: add one-time construction code here

            m_bFilesOpen = false;
            m_bBackpropThreadAbortFlag = false;
            m_bBackpropThreadsAreRunning = false;
            m_cBackprops = 0;
            m_nAfterEveryNBackprops = 1;

            m_bTestingThreadsAreRunning = false;
            m_bTestingThreadAbortFlag = false;

            m_iNextTestingPattern = 0;
            m_iNextTrainingPattern = 0;

            //::InitializeCriticalSection( &m_csTrainingPatterns );
            //::InitializeCriticalSection( &m_csTestingPatterns );

            //m_utxNeuralNet = ::CreateMutex( null, false, null );  // anonymous mutex which is unowned initially


            // allocate memory to store the distortion maps

            m_cCols = 29;
            m_cRows = 29;

            m_cCount = m_cCols * m_cRows;

            m_DispH = new double[m_cCount];
            m_DispV = new double[m_cCount];


            // create a gaussian kernel, which is constant, for use in generating elastic distortions

            int iiMid = GAUSSIAN_FIELD_SIZE / 2;  // GAUSSIAN_FIELD_SIZE is strictly odd

            double twoSigmaSquared = 2.0 * Settings.Default.m_dElasticSigma * Settings.Default.m_dElasticSigma;
            twoSigmaSquared = 1.0 / twoSigmaSquared;
            double twoPiSigma = 1.0 / Settings.Default.m_dElasticSigma
                                * Math.Sqrt(2.0 * 3.1415926535897932384626433832795);

            for (int col = 0; col < GAUSSIAN_FIELD_SIZE; ++col)
            {
                for (int row = 0; row < GAUSSIAN_FIELD_SIZE; ++row)
                {
                    m_GaussianKernel[row, col] = twoPiSigma
                                         *
                                         Math.Exp(
                                             -(((row - iiMid) * (row - iiMid) + (col - iiMid) * (col - iiMid))
                                                * twoSigmaSquared));
                }
            }
        }

        private void InitNeuralNetwork()
        {
            // TODO: add reinitialization code here
            // (SDI documents will reuse this document)

            // grab the mutex for the neural network

            //CAutoMutex tlo( m_utxNeuralNet );

            // initialize and build the neural net
            NeuralNetwork NN = m_NN; // for easier nomenclature
            NN.Initialize();

            Layer pLayer = new Layer();

            int ii, jj, kk;
            int icNeurons = 0;
            int icWeights = 0;
            double initWeight;
            string label;

            // layer zero, the input layer.
            // Create neurons: exactly the same number of neurons as the input
            // vector of 29x29=841 pixels, and no weights/connections
            pLayer = new Layer("Layer00");
            NN.Layers.Add(pLayer);

            for (ii = 0; ii < 841; ++ii)
            {
                label = String.Format("Layer00_Neuron%04d_Num%06d", ii, icNeurons);
                pLayer.Neurons.Add(new Neuron(label));
                icNeurons++;
            }

            // layer one:
            // This layer is a convolutional layer that has 6 feature maps. Each feature 
            // map is 13x13, and each unit in the feature maps is a 5x5 convolutional kernel
            // of the input layer.
            // So, there are 13x13x6 = 1014 neurons, (5x5+1)x6 = 156 weights
            pLayer = new Layer("Layer01", pLayer);
            NN.Layers.Add(pLayer);

            for (ii = 0; ii < 1014; ++ii)
            {
                label = String.Format("Layer01_Neuron%04d_Num%06d", ii, icNeurons);
                pLayer.Neurons.Add(new Neuron(label));
                icNeurons++;
            }

            for (ii = 0; ii < 156; ++ii)
            {
                label = String.Format("Layer01_Weight%04d_Num%06d", ii, icWeights);
                initWeight = 0.05 * Utils.UNIFORM_PLUS_MINUS_ONE();
                pLayer.Weights.Add(new Weight(label, initWeight));
            }

            // interconnections with previous layer: this is difficult
            // The previous layer is a top-down bitmap image that has been padded to size 29x29
            // Each neuron in this layer is connected to a 5x5 kernel in its feature map, which 
            // is also a top-down bitmap of size 13x13.  We move the kernel by TWO pixels, i.e., we
            // skip every other pixel in the input image
            int[] kernelTemplate = new int[]
                {
                    0, 1, 2, 3, 4, 
                    29, 30, 31, 32, 33, 
                    58, 59, 60, 61, 62, 
                    87, 88, 89, 90, 91, 
                    116, 117, 118, 119, 120 
                };

            int iNumWeight;
            int fm;

            for (fm = 0; fm < 6; ++fm)
            {
                for (ii = 0; ii < 13; ++ii)
                {
                    for (jj = 0; jj < 13; ++jj)
                    {
                        iNumWeight = fm * 26; // 26 is the number of weights per feature map
                        Neuron n = pLayer.Neurons[jj + ii * 13 + fm * 169];

                        n.AddConnection(UInt32.MaxValue, (uint)iNumWeight++); // bias weight

                        for (kk = 0; kk < 25; ++kk)
                        {
                            // note: max value of index == 840, corresponding to 841 neurons in previous layer
                            n.AddConnection((uint)(2 * jj + 58 * ii + kernelTemplate[kk]), (uint)iNumWeight++);
                        }
                    }
                }
            }

            // layer two:
            // This layer is a convolutional layer that has 50 feature maps.  Each feature 
            // map is 5x5, and each unit in the feature maps is a 5x5 convolutional kernel
            // of corresponding areas of all 6 of the previous layers, each of which is a 13x13 feature map
            // So, there are 5x5x50 = 1250 neurons, (5x5+1)x6x50 = 7800 weights
            pLayer = new Layer("Layer02", pLayer);
            NN.Layers.Add(pLayer);

            for (ii = 0; ii < 1250; ++ii)
            {
                label = String.Format("Layer02_Neuron%04d_Num%06d", ii, icNeurons);
                pLayer.Neurons.Add(new Neuron(label));
                icNeurons++;
            }

            for (ii = 0; ii < 7800; ++ii)
            {
                label = String.Format("Layer02_Weight%04d_Num%06d", ii, icWeights);
                initWeight = 0.05 * Utils.UNIFORM_PLUS_MINUS_ONE();
                pLayer.Weights.Add(new Weight(label, initWeight));
            }

            // Interconnections with previous layer: this is difficult
            // Each feature map in the previous layer is a top-down bitmap image whose size
            // is 13x13, and there are 6 such feature maps.  Each neuron in one 5x5 feature map of this 
            // layer is connected to a 5x5 kernel positioned correspondingly in all 6 parent
            // feature maps, and there are individual weights for the six different 5x5 kernels.  As
            // before, we move the kernel by TWO pixels, i.e., we
            // skip every other pixel in the input image.  The result is 50 different 5x5 top-down bitmap
            // feature maps
            int[] kernelTemplate2 = new int[25]
                {
                    0, 1, 2, 3, 4, 
                    13, 14, 15, 16, 17, 
                    26, 27, 28, 29, 30, 
                    39, 40, 41, 42, 43, 
                    52, 53, 54, 55, 56 
                };

            for (fm = 0; fm < 50; ++fm)
            {
                for (ii = 0; ii < 5; ++ii)
                {
                    for (jj = 0; jj < 5; ++jj)
                    {
                        iNumWeight = fm * 26; // 26 is the number of weights per feature map
                        Neuron n = pLayer.Neurons[jj + ii * 5 + fm * 25];

                        n.AddConnection(UInt32.MaxValue, (uint)iNumWeight++); // bias weight

                        for (kk = 0; kk < 25; ++kk)
                        {
                            // note: max value of index == 1013, corresponding to 1014 neurons in previous layer
                            n.AddConnection((uint)(2 * jj + 26 * ii + kernelTemplate2[kk]), (uint)iNumWeight++);
                            n.AddConnection((uint)(169 + 2 * jj + 26 * ii + kernelTemplate2[kk]), (uint)iNumWeight++);
                            n.AddConnection((uint)(338 + 2 * jj + 26 * ii + kernelTemplate2[kk]), (uint)iNumWeight++);
                            n.AddConnection((uint)(507 + 2 * jj + 26 * ii + kernelTemplate2[kk]), (uint)iNumWeight++);
                            n.AddConnection((uint)(676 + 2 * jj + 26 * ii + kernelTemplate2[kk]), (uint)iNumWeight++);
                            n.AddConnection((uint)(845 + 2 * jj + 26 * ii + kernelTemplate2[kk]), (uint)iNumWeight++);
                        }
                    }
                }
            }

            // layer three:
            // This layer is a fully-connected layer with 100 units.  Since it is fully-connected,
            // each of the 100 neurons in the layer is connected to all 1250 neurons in
            // the previous layer.
            // So, there are 100 neurons and 100*(1250+1)=125100 weights
            pLayer = new Layer("Layer03", pLayer);
            NN.Layers.Add(pLayer);

            for (ii = 0; ii < 100; ++ii)
            {
                label = String.Format("Layer03_Neuron%04d_Num%06d", ii, icNeurons);
                pLayer.Neurons.Add(new Neuron(label));
                icNeurons++;
            }

            for (ii = 0; ii < 125100; ++ii)
            {
                label = String.Format("Layer03_Weight%04d_Num%06d", ii, icWeights);
                initWeight = 0.05 * Utils.UNIFORM_PLUS_MINUS_ONE();
                pLayer.Weights.Add(new Weight(label, initWeight));
            }

            // Interconnections with previous layer: fully-connected
            iNumWeight = 0; // weights are not shared in this layer

            for (fm = 0; fm < 100; ++fm)
            {
                Neuron n = pLayer.Neurons[fm];
                n.AddConnection(UInt32.MaxValue, (uint)iNumWeight++); // bias weight

                for (ii = 0; ii < 1250; ++ii)
                {
                    n.AddConnection((uint)ii, (uint)iNumWeight++);
                }
            }

            // layer four, the final (output) layer:
            // This layer is a fully-connected layer with 10 units.  Since it is fully-connected,
            // each of the 10 neurons in the layer is connected to all 100 neurons in
            // the previous layer.
            // So, there are 10 neurons and 10*(100+1)=1010 weights
            pLayer = new Layer("Layer04", pLayer);
            NN.Layers.Add(pLayer);

            for (ii = 0; ii < 10; ++ii)
            {
                label = String.Format("Layer04_Neuron%04d_Num%06d", ii, icNeurons);
                pLayer.Neurons.Add(new Neuron(label));
                icNeurons++;
            }

            for (ii = 0; ii < 1010; ++ii)
            {
                label = String.Format("Layer04_Weight%04d_Num%06d", ii, icWeights);
                initWeight = 0.05 * Utils.UNIFORM_PLUS_MINUS_ONE();
                pLayer.Weights.Add(new Weight(label, initWeight));
            }

            // Interconnections with previous layer: fully-connected
            iNumWeight = 0; // weights are not shared in this layer

            for (fm = 0; fm < 10; ++fm)
            {
                Neuron n = pLayer.Neurons[fm];
                n.AddConnection(UInt32.MaxValue, (uint)iNumWeight++); // bias weight

                for (ii = 0; ii < 100; ++ii)
                {
                    n.AddConnection((uint)ii, (uint)iNumWeight++);
                }
            }

            //SetModifiedFlag( true );
        }

        private void button1_Click(object sender, System.EventArgs e)
        {

        }

        private bool StartBackpropagation(
            uint iStartPattern /* =0 */,
            uint iNumThreads /* =2 */,
            double initialEta /* =0.005 */,
            double minimumEta /* =0.000001 */,
            double etaDecay /* =0.990 */,
            uint nAfterEvery /* =1000 */,
            bool bDistortPatterns /* =true */,
            double estimatedCurrentMSE /* =1.0 */)
        {
            if (this.m_bBackpropThreadsAreRunning)
            {
                return false;
            }

            m_bBackpropThreadAbortFlag = false;
            m_bBackpropThreadsAreRunning = true;
            m_iNumBackpropThreadsRunning = 0;
            m_iBackpropThreadIdentifier = 0;
            m_cBackprops = iStartPattern;
            m_bNeedHessian = true;

            m_iNextTrainingPattern = iStartPattern;
            //m_hWndForBackpropPosting = hWnd;

            if (m_iNextTrainingPattern < 0)
            {
                m_iNextTrainingPattern = 0;
            }
            if (m_iNextTrainingPattern >= Settings.Default.m_nItemsTrainingImages)
            {
                m_iNextTrainingPattern = (uint)(Settings.Default.m_nItemsTrainingImages - 1);
            }

            if (iNumThreads < 1)
            {
                iNumThreads = 1;
            }

            if (iNumThreads > 10) // 10 is arbitrary upper limit
            {
                iNumThreads = 10;
            }

            this.m_NN.EtaLearningRate = initialEta;
            this.m_NN.EtaLearningRatePrevious = initialEta;
            m_dMinimumEta = minimumEta;
            m_dEtaDecay = etaDecay;
            m_nAfterEveryNBackprops = nAfterEvery;
            m_bDistortTrainingPatterns = bDistortPatterns;

            m_dEstimatedCurrentMSE = estimatedCurrentMSE;
            // estimated number that will define whether a forward calculation's error is significant enough to warrant backpropagation

            RandomizeTrainingPatternSequence();

            for (uint ii = 0; ii < iNumThreads; ++ii)
            {
                Thread pThread = new Thread(BackpropagationThread);
                pThread.Start(this);

                m_pBackpropThreads[ii] = pThread;
                m_iNumBackpropThreadsRunning++;
            }

            return true;
        }

        private void RandomizeTrainingPatternSequence()
        {
            // randomizes the order of m_iRandomizedTrainingPatternSequence, which is a uint array
            // holding the numbers 0..59999 in random order

            // CAutoCS tlo( m_csTrainingPatterns );
            uint ii, jj, iiMax, iiTemp;

            iiMax = (uint)Settings.Default.m_nItemsTrainingImages;

            for (ii = 0; ii < iiMax; ++ii)
            {
                m_iRandomizedTrainingPatternSequence[ii] = ii;
            }


            // now at each position, swap with a random position

            for (ii = 0; ii < iiMax; ++ii)
            {
                jj = (uint)((new Random()).NextDouble() * iiMax);

                iiTemp = m_iRandomizedTrainingPatternSequence[ii];
                m_iRandomizedTrainingPatternSequence[ii] = m_iRandomizedTrainingPatternSequence[jj];
                m_iRandomizedTrainingPatternSequence[jj] = iiTemp;
            }

        }

        private void BackpropagationThread(object pVoid)
        {
            MainForm pThis = (MainForm)pVoid;

            double[] inputVector = new double[841]; // note: 29x29, not 28x28
            double[] targetOutputVector = new double[10];
            double[] actualOutputVector = new double[10];
            byte[] grayLevels = new byte[Settings.Default.m_nRowsImages * Settings.Default.m_nRowsImages];
            int label = 0;

            List<List<double>> memorizedNeuronOutputs = new List<List<double>>();

            while (pThis.m_bBackpropThreadAbortFlag == false)
            {
                uint iSequentialNum;
                int iRet = (int)pThis.GetNextTrainingPattern(grayLevels, label, true, true, out iSequentialNum);

                if (label < 0)
                {
                    label = 0;
                }

                if (label > 9)
                {
                    label = 9;
                }

                // post message to the dialog, telling it which pattern this thread is currently working on

                //if (pThis.m_hWndForBackpropPosting != null)
                {
                    //::PostMessage( pThis.m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 1L, (LPARAM)iSequentialNum );
                }


                // pad to 29x29, convert to double precision

                int ii;
                for (ii = 0; ii < 841; ++ii)
                {
                    inputVector[ii] = 1.0; // one is white, -one is black
                }

                // top row of inputVector is left as zero, left-most column is left as zero 

                for (ii = 0; ii < Settings.Default.m_nRowsImages; ++ii)
                {
                    int jj;
                    for (jj = 0; jj < Settings.Default.m_nColsImages; ++jj)
                    {
                        inputVector[1 + jj + 29 * (ii + 1)] =
                            grayLevels[jj + Settings.Default.m_nRowsImages * ii] / 128.0 - 1.0;
                        // one is white, -one is black
                    }
                }

                // desired output vector
                for (ii = 0; ii < 10; ++ii)
                {
                    targetOutputVector[ii] = -1.0;
                }

                targetOutputVector[label] = 1.0;

                // now backpropagate
                pThis.BackpropagateNeuralNet(
                    inputVector,
                    841,
                    targetOutputVector,
                    actualOutputVector,
                    10,
                    memorizedNeuronOutputs,
                    pThis.m_bDistortTrainingPatterns);


                // calculate error for this pattern and post it to the hwnd so it can calculate a running 
                // estimate of MSE
                double dMSE = 0.0;
                for (ii = 0; ii < 10; ++ii)
                {
                    dMSE += (actualOutputVector[ii] - targetOutputVector[ii])
                            * (actualOutputVector[ii] - targetOutputVector[ii]);
                }

                dMSE /= 2.0;

                uint scaledMSE = (uint)(Math.Sqrt(dMSE) * 2.0e8);
                // arbitrary large pre-agreed upon scale factor; taking sqrt is simply to improve the scaling

                //if (pThis.m_hWndForBackpropPosting != null)
                {
                    //::PostMessage( pThis.m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 2L, (LPARAM)scaledMSE );
                }


                // determine the neural network's answer, and compare it to the actual answer.
                // Post a message if the answer was incorrect, so the dialog can display mis-recognition
                // statistics

                int iBestIndex = 0;
                double maxValue = -99.0;

                for (ii = 0; ii < 10; ++ii)
                {
                    if (actualOutputVector[ii] > maxValue)
                    {
                        iBestIndex = ii;
                        maxValue = actualOutputVector[ii];
                    }
                }

                if (iBestIndex != label)
                {
                    // pattern was mis-recognized.  Notify the testing dialog
                    //if (pThis.m_hWndForBackpropPosting != null)
                    {
                        //::PostMessage( pThis.m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 8L, (LPARAM)0L );
                    }
                }

            } // end of main "while not abort flag" loop
        }

        private void BackpropagateNeuralNet(
            double[] inputVector,
            int iCount,
            double[] targetOutputVector,
            double[] actualOutputVector,
            int oCount,
            List<List<double>> pMemorizedNeuronOutputs,
            bool bDistort)
        {
            bool bWorthwhileToBackpropagate; /////// part of code review
            {
                if (((m_cBackprops % m_nAfterEveryNBackprops) == 0) && (m_cBackprops != 0))
                {
                    double eta = this.m_NN.EtaLearningRate;
                    eta *= m_dEtaDecay;
                    if (eta < m_dMinimumEta)
                    {
                        eta = m_dMinimumEta;
                    }

                    this.m_NN.EtaLearningRatePrevious = this.m_NN.EtaLearningRate;
                    this.m_NN.EtaLearningRate = eta;
                }


                if (this.m_bNeedHessian || ((m_cBackprops % Settings.Default.m_nItemsTrainingImages) == 0))
                {
                    // adjust the Hessian.  This is a lengthy operation, since it must process approx 500 labels
                    CalculateHessian();

                    m_bNeedHessian = false;
                }


                // determine if it's time to randomize the sequence of training patterns (currently once per epoch)
                if ((m_cBackprops % Settings.Default.m_nItemsTrainingImages) == 0)
                {
                    RandomizeTrainingPatternSequence();
                }

                // increment counter for tracking number of backprops
                m_cBackprops++;

                // forward calculate through the neural net
                CalculateNeuralNet(inputVector, iCount, actualOutputVector, oCount, pMemorizedNeuronOutputs, bDistort);

                // calculate error in the output of the neural net
                // note that this code duplicates that found in many other places, and it's probably sensible to 
                // define a (global/static ??) function for it
                double dMSE = 0.0;
                for (int ii = 0; ii < 10; ++ii)
                {
                    dMSE += (actualOutputVector[ii] - targetOutputVector[ii])
                            * (actualOutputVector[ii] - targetOutputVector[ii]);
                }

                dMSE /= 2.0;
                bWorthwhileToBackpropagate = dMSE > (0.10 * this.m_dEstimatedCurrentMSE);

                if (bWorthwhileToBackpropagate && (pMemorizedNeuronOutputs == null))
                {
                    // the caller has not provided a place to store neuron outputs, so we need to
                    // backpropagate now, while the neural net is still captured.  Otherwise, another thread
                    // might come along and call CalculateNeuralNet(), which would entirely change the neuron
                    // outputs and thereby inject errors into backpropagation 
                    m_NN.Backpropagate(actualOutputVector, targetOutputVector, (uint)oCount, null);

                    //SetModifiedFlag(true);

                    // we're done, so return
                    return;
                }
            }

            // if we have reached here, then the mutex for the neural net has been released for other 
            // threads.  The caller must have provided a place to store neuron outputs, which we can 
            // use to backpropagate, even if other threads call CalculateNeuralNet() and change the outputs
            // of the neurons
            if (bWorthwhileToBackpropagate)
            {
                m_NN.Backpropagate(actualOutputVector, targetOutputVector, (uint)oCount, pMemorizedNeuronOutputs);

                // set modified flag to prevent closure of doc without a warning

                //SetModifiedFlag(true);
            }
        }

        private void CalculateHessian()
        {
            // controls the Neural network's calculation if the diagonal Hessian for the Neural net
            // This will be called from a thread, so although the calculation is lengthy, it should not interfere
            // with the UI

            // we need the neural net exclusively during this calculation, so grab it now
            double[] inputVector = new double[841];
            double[] targetOutputVector = new double[10];
            double[] actualOutputVector = new double[10];

            byte[] grayLevels = new byte[g_cImageSize * g_cImageSize];
            int label = 0;
            uint kk;

            // calculate the diagonal Hessian using 500 random patterns, per Yann LeCun 1998 "Gradient-Based Learning
            // Applied To Document Recognition"

            // message to dialog that we are commencing calculation of the Hessian

            //if (m_hWndForBackpropPosting != null)
            {
                // wParam == 4L . related to Hessian, lParam == 1L . commenced calculation
                //::PostMessage( m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 4L, 1L );
            }

            // some of this code is similar to the BackpropagationThread() code

            m_NN.EraseHessianInformation();

            uint numPatternsSampled = (uint)Settings.Default.m_nNumHessianPatterns;

            for (kk = 0; kk < numPatternsSampled; ++kk)
            {
                GetRandomTrainingPattern(grayLevels, label, true);

                if (label < 0)
                {
                    label = 0;
                }

                if (label > 9)
                {
                    label = 9;
                }

                // pad to 29x29, convert to double precision
                int ii;
                for (ii = 0; ii < 841; ++ii)
                {
                    inputVector[ii] = 1.0; // one is white, -one is black
                }

                // top row of inputVector is left as zero, left-most column is left as zero 
                for (ii = 0; ii < g_cImageSize; ++ii)
                {
                    int jj;
                    for (jj = 0; jj < g_cImageSize; ++jj)
                    {
                        inputVector[1 + jj + 29 * (ii + 1)] = grayLevels[jj + g_cImageSize * ii] / 128.0 - 1.0; // one is white, -one is black
                    }
                }

                // desired output vector
                for (ii = 0; ii < 10; ++ii)
                {
                    targetOutputVector[ii] = -1.0;
                }

                targetOutputVector[label] = 1.0;


                // apply distortion map to inputVector.  It's not certain that this is needed or helpful.
                // The second derivatives do NOT rely on the output of the neural net (i.e., because the 
                // second derivative of the MSE function is exactly 1 (one), regardless of the actual output
                // of the net).  However, since the backpropagated second derivatives rely on the outputs of
                // each neuron, distortion of the pattern might reveal previously-unseen information about the
                // nature of the Hessian.  But I am reluctant to give the full distortion, so I set the
                // severityFactor to only 2/3 approx

                GenerateDistortionMap(0.65);
                ApplyDistortionMap(inputVector);


                // forward calculate the neural network

                m_NN.Calculate(inputVector, 841, actualOutputVector, 10, null);


                // backpropagate the second derivatives

                m_NN.BackpropagateSecondDervatives(actualOutputVector, targetOutputVector, 10);


                // progress message to dialog that we are calculating the Hessian

                if (kk % 50 == 0)
                {
                    // every 50 iterations ...
                    //if (m_hWndForBackpropPosting != null)
                    {
                        // wParam == 4L . related to Hessian, lParam == 2L . progress indicator
                        //::PostMessage( m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 4L, 2L );
                    }
                }

                if (this.m_bBackpropThreadAbortFlag)
                {
                    break;
                }
            }

            m_NN.DivideHessianInformationBy(numPatternsSampled);

            // message to dialog that we are finished calculating the Hessian

            //if (m_hWndForBackpropPosting != null)
            {
                // wParam == 4L . related to Hessian, lParam == 4L . finished calculation
                //::PostMessage( m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 4L, 4L );
            }

        }

        uint GetRandomTrainingPattern(byte[] pArray /* =null */, int pLabel /* =null */, bool bFlipGrayscale /* =true */ )
        {
            // returns the number of the pattern corresponding to the pattern stored in pArray

            //CAutoCS tlo( m_csTrainingPatterns );

            uint patternNum = (uint)(Utils.UNIFORM_ZERO_THRU_ONE() * (Settings.Default.m_nItemsTrainingImages - 1));

            GetTrainingPatternArrayValues((int)patternNum, pArray, ref pLabel, bFlipGrayscale);

            return patternNum;
        }

        private void CalculateNeuralNet(
            double[] inputVector,
            int count,
            double[] outputVector = null,
            int oCount = 0,
            List<List<double>> pNeuronOutputs = null,
            bool bDistort = false)
        {
            // wrapper function for neural net's Calculate() function, needed because the NN is a protected member
            // waits on the neural net mutex (using the CAutoMutex object, which automatically releases the
            // mutex when it goes out of scope) so as to restrict access to one thread at a time
            if (bDistort)
            {
                GenerateDistortionMap();
                ApplyDistortionMap(inputVector);
            }

            m_NN.Calculate(inputVector, (uint)count, outputVector, (uint)oCount, pNeuronOutputs);
        }

        private void GenerateDistortionMap(double severityFactor = 1.0)
        {
            // generates distortion maps in each of the horizontal and vertical directions
            // Three distortions are applied: a scaling, a rotation, and an elastic distortion
            // Since these are all linear transformations, we can simply add them together, after calculation
            // one at a time

            // The input parameter, severityFactor, let's us control the severity of the distortions relative
            // to the default values.  For example, if we only want half as harsh a distortion, set
            // severityFactor == 0.5

            // First, elastic distortion, per Patrice Simard, "Best Practices For Convolutional Neural Networks..."
            // at page 2.
            // Three-step process: seed array with uniform randoms, filter with a Gaussian kernel, normalize (scale)
            int row, col;
            double[] uniformH = new double[m_cCount];
            double[] uniformV = new double[m_cCount];


            for (col = 0; col < m_cCols; ++col)
            {
                for (row = 0; row < m_cRows; ++row)
                {
                    this.AtAssign(uniformH, row, col, Utils.UNIFORM_PLUS_MINUS_ONE());
                    AtAssign(uniformV, row, col, Utils.UNIFORM_PLUS_MINUS_ONE());
                }
            }

            // filter with gaussian
            double elasticScale = severityFactor * Settings.Default.m_dElasticScaling;
            int iiMid = GAUSSIAN_FIELD_SIZE / 2; // GAUSSIAN_FIELD_SIZE is strictly odd

            for (col = 0; col < m_cCols; ++col)
            {
                for (row = 0; row < m_cRows; ++row)
                {
                    double fConvolvedH = 0.0;
                    double fConvolvedV = 0.0;

                    int xxx;
                    for (xxx = 0; xxx < GAUSSIAN_FIELD_SIZE; ++xxx)
                    {
                        int yyy;
                        for (yyy = 0; yyy < GAUSSIAN_FIELD_SIZE; ++yyy)
                        {
                            int xxxDisp = col - iiMid + xxx;
                            int yyyDisp = row - iiMid + yyy;

                            double fSampleH;
                            double fSampleV;
                            if (xxxDisp < 0 || xxxDisp >= m_cCols || yyyDisp < 0 || yyyDisp >= m_cRows)
                            {
                                fSampleH = 0.0;
                                fSampleV = 0.0;
                            }
                            else
                            {
                                fSampleH = At(uniformH, yyyDisp, xxxDisp);
                                fSampleV = At(uniformV, yyyDisp, xxxDisp);
                            }

                            fConvolvedH += fSampleH * m_GaussianKernel[yyy, xxx];
                            fConvolvedV += fSampleV * m_GaussianKernel[yyy, xxx];
                        }
                    }

                    this.AtAssign(m_DispH, row, col, elasticScale * fConvolvedH);
                    this.AtAssign(m_DispV, row, col, elasticScale * fConvolvedV);
                }
            }

            // next, the scaling of the image by a random scale factor
            // Horizontal and vertical directions are scaled independently
            double dSFHoriz = severityFactor * Settings.Default.m_dMaxScaling / 100.0 * Utils.UNIFORM_PLUS_MINUS_ONE();
            // m_dMaxScaling is a percentage
            double dSFVert = severityFactor * Settings.Default.m_dMaxScaling / 100.0 * Utils.UNIFORM_PLUS_MINUS_ONE();
            // m_dMaxScaling is a percentage


            int iMid = m_cRows / 2;
            for (row = 0; row < m_cRows; ++row)
            {
                for (col = 0; col < m_cCols; ++col)
                {
                    AtAssign(m_DispH, row, col, At(m_DispH, row, col) + dSFHoriz * (col - iMid));
                    AtAssign(m_DispV, row, col, At(m_DispH, row, col) - (dSFVert * (iMid - row))); // negative because of top-down bitmap
                }
            }


            // finally, apply a rotation
            double angle = severityFactor * Settings.Default.m_dMaxRotation * Utils.UNIFORM_PLUS_MINUS_ONE();
            angle = angle * 3.1415926535897932384626433832795 / 180.0; // convert from degrees to radians

            double cosAngle = Math.Cos(angle);
            double sinAngle = Math.Sin(angle);

            for (row = 0; row < m_cRows; ++row)
            {
                for (col = 0; col < m_cCols; ++col)
                {
                    AtAssign(m_DispH, row, col, this.At(m_DispH, row, col) + (col - iMid) * (cosAngle - 1) - (iMid - row) * sinAngle);
                    AtAssign(m_DispV, row, col, this.At(m_DispH, row, col) - ((iMid - row) * (cosAngle - 1) + (col - iMid) * sinAngle));
                    // negative because of top-down bitmap
                }
            }

        }

        private void ApplyDistortionMap(double[] inputVector)
        {
            // applies the current distortion map to the input vector

            // For the mapped array, we assume that 0.0 == background, and 1.0 == full intensity information
            // This is different from the input vector, in which +1.0 == background (white), and 
            // -1.0 == information (black), so we must convert one to the other
            List<List<double>> mappedVector = new List<List<double>>(m_cRows);
            for (int i = 0; i < m_cRows; ++i)
            {
                mappedVector.Add(new List<double>(m_cCols));
            }

            int row, col;

            for (row = 0; row < m_cRows; ++row)
            {
                for (col = 0; col < m_cCols; ++col)
                {
                    // the pixel at sourceRow, sourceCol is an "phantom" pixel that doesn't really exist, and
                    // whose value must be manufactured from surrounding real pixels (i.e., since 
                    // sourceRow and sourceCol are floating point, not ints, there's not a real pixel there)
                    // The idea is that if we can calculate the value of this phantom pixel, then its 
                    // displacement will exactly fit into the current pixel at row, col (which are both ints)
                    double sourceRow = row - this.At(this.m_DispV, row, col);
                    double sourceCol = col - this.At(this.m_DispH, row, col);

                    // weights for bi-linear interpolation

                    double fracRow = sourceRow - (int)sourceRow;
                    double fracCol = sourceCol - (int)sourceCol;


                    double w1 = (1.0 - fracRow) * (1.0 - fracCol);
                    double w2 = (1.0 - fracRow) * fracCol;
                    double w3 = fracRow * (1 - fracCol);
                    double w4 = fracRow * fracCol;

                    // limit indexes

                    /*
                                while (sourceRow >= m_cRows ) sourceRow -= m_cRows;
                                while (sourceRow < 0 ) sourceRow += m_cRows;
			
                                while (sourceCol >= m_cCols ) sourceCol -= m_cCols;
                                while (sourceCol < 0 ) sourceCol += m_cCols;
                    */
                    bool bSkipOutOfBounds = false;

                    if ((sourceRow + 1.0) >= m_cRows)
                    {
                        bSkipOutOfBounds = true;
                    }

                    if (sourceRow < 0)
                    {
                        bSkipOutOfBounds = true;
                    }

                    if ((sourceCol + 1.0) >= m_cCols)
                    {
                        bSkipOutOfBounds = true;
                    }

                    if (sourceCol < 0)
                    {
                        bSkipOutOfBounds = true;
                    }

                    double sourceValue;
                    if (bSkipOutOfBounds == false)
                    {
                        // the supporting pixels for the "phantom" source pixel are all within the 
                        // bounds of the character grid.
                        // Manufacture its value by bi-linear interpolation of surrounding pixels

                        int sRow = (int)sourceRow;
                        int sCol = (int)sourceCol;

                        int sRowp1 = sRow + 1;
                        int sColp1 = sCol + 1;

                        while (sRowp1 >= m_cRows)
                        {
                            sRowp1 -= m_cRows;
                        }

                        while (sRowp1 < 0)
                        {
                            sRowp1 += m_cRows;
                        }

                        while (sColp1 >= m_cCols)
                        {
                            sColp1 -= m_cCols;
                        }

                        while (sColp1 < 0)
                        {
                            sColp1 += m_cCols;
                        }

                        // perform bi-linear interpolation
                        sourceValue = w1 * At(inputVector, sRow, sCol) + w2 * At(inputVector, sRow, sColp1)
                                      + w3 * At(inputVector, sRowp1, sCol) + w4 * At(inputVector, sRowp1, sColp1);
                    }
                    else
                    {
                        // At least one supporting pixel for the "phantom" pixel is outside the
                        // bounds of the character grid. Set its value to "background"
                        sourceValue = 1.0; // "background" color in the -1 . +1 range of inputVector
                    }

                    mappedVector[row][col] = 0.5 * (1.0 - sourceValue);
                    // conversion to 0.1 range we are using for mappedVector
                }
            }

            // now, invert again while copying back into original vector
            for (row = 0; row < m_cRows; ++row)
            {
                for (col = 0; col < m_cCols; ++col)
                {
                    this.AtAssign(inputVector, row, col, 1.0 - 2.0 * mappedVector[row][col]);
                }
            }
        }

        private void AtAssign(double[] p, int row, int col, double newValue) // zero-based indices, starting at bottom-left
        {
            int location = row * m_cCols + col;
            p[location] = newValue;
        }

        private double At(double[] p, int row, int col) // zero-based indices, starting at bottom-left
        {
            int location = row * m_cCols + col;
            return p[location];
        }

        uint GetNextTrainingPattern(byte[] pArray /* =null */, int pLabel /* =null */,
                                           bool bFlipGrayscale /* =true */, bool bFromRandomizedPatternSequence /* =true */,
                                           out uint iSequenceNum /* =null */)
        {
            // returns the number of the pattern corresponding to the pattern that will be stored in pArray
            // if bool bFromRandomizedPatternSequence is true (which is the default) then the pattern
            // stored will be a pattern from the randomized sequence; otherwise the pattern will be a straight
            // sequential run through all the training patterns, from start to finish.  The sequence number,
            // which runs from 0..59999 monotonically, is returned in iSequenceNum (if it's not null)

            // CAutoCS tlo( m_csTrainingPatterns );

            uint iPatternNum;

            if (bFromRandomizedPatternSequence == false)
            {
                iPatternNum = m_iNextTrainingPattern;
            }
            else
            {
                iPatternNum = m_iRandomizedTrainingPatternSequence[m_iNextTrainingPattern];
            }


            GetTrainingPatternArrayValues((int)iPatternNum, pArray, ref pLabel, bFlipGrayscale);

            iSequenceNum = m_iNextTrainingPattern;

            m_iNextTrainingPattern++;

            if (m_iNextTrainingPattern >= Settings.Default.m_nItemsTrainingImages)
            {
                m_iNextTrainingPattern = 0;
            }

            return iPatternNum;
        }

        void GetTrainingPatternArrayValues(int iNumImage /* =0 */, byte[] pArray /* =null */, ref int pLabel /* =null */,
                                              bool bFlipGrayscale /* =true */ )
        {
            // fills an unsigned char array with gray values, corresponding to iNumImage, and also
            // returns the label for the image

            //CAutoCS tlo( m_csTrainingPatterns );

            int cCount = (int)(g_cImageSize * g_cImageSize);
            int fPos;

            if (true)// m_bFilesOpen != false )
            {
                if (pArray != null)
                {
                    fPos = 16 + iNumImage * cCount;  // 16 compensates for file header info
                    //m_fileTrainingImages.Seek( fPos, CFile::begin );
                    //m_fileTrainingImages.Read( pArray, cCount );

                    if (bFlipGrayscale != false)
                    {
                        for (int ii = 0; ii < cCount; ++ii)
                        {
                            pArray[ii] = (byte)(255 - pArray[ii]);
                        }
                    }
                }

                if (pLabel != null)
                {
                    fPos = 8 + iNumImage;
                    char r = ' ';
                    //m_fileTrainingLabels.Seek( fPos, CFile::begin );
                    //m_fileTrainingLabels.Read( r, 1 );  // single byte

                    pLabel = r;
                }
            }
            else  // no files are open: return a simple gray wedge
            {
                if (pArray != null)
                {
                    for (int ii = 0; ii < cCount; ++ii)
                    {
                        pArray[ii] = (byte)(ii * 255 / cCount);
                    }
                }

                if (pLabel != null)
                {
                    pLabel = Int32.MaxValue;
                }
            }
        }

        double GetCurrentEta()
        {
            return this.m_NN.EtaLearningRate;
        }

        double GetPreviousEta()
        {
            // provided because threads might change the current eta before we are able to read it
            return m_NN.EtaLearningRatePrevious;
        }

        uint GetCurrentTrainingPatternNumber(bool bFromRandomizedPatternSequence /* =false */ )
        {
            // returns the current number of the training pattern, either from the straight sequence, or from
            // the randomized sequence
            uint iRet;

            if (bFromRandomizedPatternSequence == false)
            {
                iRet = m_iNextTrainingPattern;
            }
            else
            {
                iRet = m_iRandomizedTrainingPatternSequence[m_iNextTrainingPattern];
            }

            return iRet;
        }

        uint GetNextTestingPatternNumber()
        {
            return (uint)this.m_iNextTestingPattern;
        }

        uint GetNextTestingPattern(byte[] pArray /* =null */, ref int pLabel /* =null */, bool bFlipGrayscale /* =true */ )
        {
            // returns the number of the pattern corresponding to the pattern stored in pArray

            //CAutoCS tlo( m_csTestingPatterns );


            GetTestingPatternArrayValues(m_iNextTestingPattern, pArray, ref pLabel, bFlipGrayscale);

            uint iRet = (uint)this.m_iNextTestingPattern;
            m_iNextTestingPattern++;

            if (m_iNextTestingPattern >= Settings.Default.m_nItemsTestingImages)
            {
                m_iNextTestingPattern = 0;
            }

            return iRet;
        }

        void GetTestingPatternArrayValues(int iNumImage /* =0 */, byte[] pArray /* =null */, ref int pLabel /* =null */,
                                             bool bFlipGrayscale /* =true */ )
        {
            // fills an unsigned char array with gray values, corresponding to iNumImage, and also
            // returns the label for the image

            //CAutoCS tlo( m_csTestingPatterns );

            int cCount = (int)(g_cImageSize * g_cImageSize);
            int fPos;

            if (true)// m_bFilesOpen != false )
            {
                if (pArray != null)
                {
                    fPos = 16 + iNumImage * cCount;  // 16 compensates for file header info
                    //m_fileTestingImages.Seek( fPos, CFile::begin );
                    //m_fileTestingImages.Read( pArray, cCount );

                    if (bFlipGrayscale != false)
                    {
                        for (int ii = 0; ii < cCount; ++ii)
                        {
                            pArray[ii] = (byte)(255 - pArray[ii]);
                        }
                    }
                }

                if (pLabel != null)
                {
                    fPos = 8 + iNumImage;
                    char r = '\0';
                    //m_fileTestingLabels.Seek( fPos, CFile::begin );
                    //m_fileTestingLabels.Read( &r, 1 );  // single byte

                    pLabel = r;
                }
            }
            else  // no files are open: return a simple gray wedge
            {
                if (pArray != null)
                {
                    for (int ii = 0; ii < cCount; ++ii)
                    {
                        pArray[ii] = (byte)(ii * 255 / cCount);
                    }
                }

                if (pLabel != null)
                {
                    pLabel = (int)Int32.MaxValue;
                }
            }
        }

        bool StartTesting(uint iStartingPattern, uint iNumThreads, bool bDistortPatterns,
                             uint iWhichImageSet /* =1 */ )
        {
            // creates and starts testing threads

            if (m_bTestingThreadsAreRunning != false)
                return false;

            m_bTestingThreadAbortFlag = false;
            m_bTestingThreadsAreRunning = true;
            m_iNumTestingThreadsRunning = 0;
            m_iTestingThreadIdentifier = 0;

            m_iNextTestingPattern = (int)iStartingPattern;
            //m_hWndForTestingPosting = hWnd;
            m_iWhichImageSet = iWhichImageSet;

            if (m_iWhichImageSet > 1)
                m_iWhichImageSet = 1;
            if (m_iWhichImageSet < 0)  // which is not possible, since m_iWhichImageSet is a uint
                m_iWhichImageSet = 0;

            if (m_iNextTestingPattern < 0)
                m_iNextTestingPattern = 0;
            if (m_iNextTestingPattern >= Settings.Default.m_nItemsTestingImages)
                m_iNextTestingPattern = (int)(Settings.Default.m_nItemsTestingImages - 1);

            if (iNumThreads < 1)
                iNumThreads = 1;
            if (iNumThreads > 10)  // 10 is arbitrary upper limit
                iNumThreads = 10;

            m_bDistortTestingPatterns = bDistortPatterns;

            for (uint ii = 0; ii < iNumThreads; ++ii)
            {
                Thread pThread = new Thread(TestingThread);
                pThread.Start(this);

                m_pTestingThreads[ii] = pThread;
                m_iNumTestingThreadsRunning++;
            }

            return true;
        }

        private void TestingThread(object pVoid)
        {
            // thread for testing of Neural net
            // Continuously get the doc's next pattern, puts it through the neural net, and
            // inspects the output.  As the thread goes through the patterns, it post messages to the
            // m_hWndForTestingPosting, which presumably is the dialog that shows testing results,
            // advising it of the current pattern being tested.  If the actual output from the 
            // neural net differs from the desired output, another message is posted, advising the 
            // m_hWndForTestingPosting of the identity of the mis-recognized pattern

            // thread is owned by the doc and accepts a pointer to the doc as a parameter


            MainForm pThis = (MainForm)pVoid;

            // set thread name (helps during debugging)

            char[] str = new char[25];  // must use chars, not TCHARs, for SetThreadname function
            //sprintf( str, "TEST%02d", pThis.m_iTestingThreadIdentifier++ );
            //SetThreadName( -1, str );

            // do the work

            double[] inputVector = new double[841];
            double[] targetOutputVector = new double[10];
            double[] actualOutputVector = new double[10];
            double dPatternMSE = 0.0;
            double dTotalMSE = 0.0;
            uint scaledMSE = 0;
            uint iPatternsProcessed = 0;

            byte[] grayLevels = new byte[g_cImageSize * g_cImageSize];
            int label = 0;
            int ii, jj;
            uint iPatNum, iSequentialNum;

            while (pThis.m_bTestingThreadAbortFlag == false)
            {
                // testing image set or training image set

                if (pThis.m_iWhichImageSet == 1)
                {
                    // testing set

                    iPatNum = pThis.GetNextTestingPattern(grayLevels, ref label, true);

                    // post message to the dialog, telling it which pattern this thread is currently working on

                    if (true)// pThis.m_hWndForTestingPosting != null )
                    {
                        //::PostMessage( pThis.m_hWndForTestingPosting, UWM_TESTING_NOTIFICATION, 1L, (LPARAM)iPatNum );
                    }
                }
                else
                {
                    // training set

                    iPatNum = pThis.GetNextTrainingPattern(grayLevels, label, true, false, out iSequentialNum);

                    // post message to the dialog, telling it which pattern this thread is currently working on

                    if (true)// pThis.m_hWndForTestingPosting != null )
                    {
                        //::PostMessage( pThis.m_hWndForTestingPosting, UWM_TESTING_NOTIFICATION, 1L, (LPARAM)iSequentialNum );
                    }
                }

                if (label < 0)
                {
                    label = 0;
                }
                if (label > 9)
                {
                    label = 9;
                }

                // pad to 29x29, convert to double precision
                for (ii = 0; ii < 841; ++ii)
                {
                    inputVector[ii] = 1.0;  // one is white, -one is black
                }

                // top row of inputVector is left as zero, left-most column is left as zero 

                for (ii = 0; ii < g_cImageSize; ++ii)
                {
                    for (jj = 0; jj < g_cImageSize; ++jj)
                    {
                        inputVector[1 + jj + 29 * (ii + 1)] = (double)((int)(byte)grayLevels[jj + g_cImageSize * ii]) / 128.0 - 1.0;
                        // one is white, -one is black
                    }
                }

                // desired output vector

                for (ii = 0; ii < 10; ++ii)
                {
                    targetOutputVector[ii] = -1.0;
                }
                targetOutputVector[label] = 1.0;


                // now calculate output of neural network

                pThis.CalculateNeuralNet(inputVector, 841, actualOutputVector, 10, null, pThis.m_bDistortTestingPatterns);


                // calculate error for this pattern and accumulate it for posting of
                // total MSE of all patterns when thread is exiting

                dPatternMSE = 0.0;
                for (ii = 0; ii < 10; ++ii)
                {
                    dPatternMSE += (actualOutputVector[ii] - targetOutputVector[ii]) * (actualOutputVector[ii] - targetOutputVector[ii]);
                }

                dPatternMSE /= 2.0;

                dTotalMSE += dPatternMSE;
                ++iPatternsProcessed;


                // determine the neural network's answer, and compare it to the actual answer

                int iBestIndex = 0;
                double maxValue = -99.0;
                uint code;

                for (ii = 0; ii < 10; ++ii)
                {
                    if (actualOutputVector[ii] > maxValue)
                    {
                        iBestIndex = ii;
                        maxValue = actualOutputVector[ii];
                    }
                }


                // moment of truth: Did neural net get the correct answer

                if (iBestIndex != label)
                {
                    // pattern was mis-recognized.  Notify the testing dialog

                    // lParam is built to contain a coded bit pattern, as follows:
                    //
                    //  0          1          2         3
                    //  0123456 7890123 456789012345678901
                    // |  act  |  tar  |    pattern num   |
                    //
                    // where act == actual output of the neural net, and tar == target
                    // this gives 2^7 = 128 possible outputs (only 10 are needed here... future expansion??)
                    // and 2^18 = 262144 possible pattern numbers ( only 10000 are needed here )

                    //code = (iPatNum & 0x0003FFFF);
                    //code |= (label & 0x0000007F) << 18;
                    //code |= (iBestIndex & 0x0000007F) << 25;

                    if (true)// pThis.m_hWndForTestingPosting != null )
                    {
                        //::PostMessage( pThis.m_hWndForTestingPosting, UWM_TESTING_NOTIFICATION, 2L, (LPARAM)code );
                    }
                }

            }


            // post the total MSE of tested patterns to the hwnd

            double divisor = (double)((iPatternsProcessed > 1) ? iPatternsProcessed : 1);
            dTotalMSE /= divisor;
            scaledMSE = (uint)(Math.Sqrt(dTotalMSE) * 2.0e8);  // arbitrary large pre-agreed upon scale factor; taking sqrt is simply to improve the scaling

            if (true)// pThis.m_hWndForTestingPosting != null )
            {
                //::PostMessage( pThis.m_hWndForTestingPosting, UWM_TESTING_NOTIFICATION, 4L, (LPARAM)scaledMSE );
            }
        }
    }
}
