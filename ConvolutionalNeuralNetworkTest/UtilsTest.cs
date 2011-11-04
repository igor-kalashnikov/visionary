using System;
using NUnit.Framework;

namespace Visionary.ConvolutionalNeuralNetworkTest
{
    using Visionary.ConvolutionalNeuralNetwork;

    [TestFixture]
    internal class UtilsTest
    {
        [Test]
        public void SigmoidTest()
        {
            double x = 9.9;
            Assert.That(1.7159 * Math.Tanh(0.66666667 * x), Is.EqualTo(Utils.Sigmoid(x)));
            x = -3.9;
            Assert.That(1.7159 * Math.Tanh(0.66666667 * x), Is.EqualTo(Utils.Sigmoid(x)));
            x = 0;
            Assert.That(1.7159 * Math.Tanh(0.66666667 * x), Is.EqualTo(Utils.Sigmoid(x)));
        }

        [Test]
        public void SigmoidDerivativeTest()
        {
            double x = 9.9;
            Assert.That(
                0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x), Is.EqualTo(Utils.SigmoidDerivative(x)));
            x = -3.9;
            Assert.That(
                0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x), Is.EqualTo(Utils.SigmoidDerivative(x)));
            x = 0.0;
            Assert.That(
                0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x), Is.EqualTo(Utils.SigmoidDerivative(x)));
        }

        [Test]
        public void RandomDoubleTest()
        {
            for (int i = 0; i < 1000; ++i)
            {
                double d = Utils.RandomDouble(0.0, 1.0);
                Assert.That(d, Is.InRange(0.0, 1.0));
                d = Utils.RandomDouble(0.99999999, 1.0);
                Assert.That(d, Is.InRange(0.9999999, 1.0));
                d = Utils.RandomDouble(0.0, 1.0);
                Assert.That(d, Is.InRange(0.0, 1.0));
                d = Utils.RandomDouble(-1.0, 1.0);
                Assert.That(d, Is.InRange(-1.0, 1.0));
                d = Utils.RandomDouble(-10000.0, 10000.0);
                Assert.That(d, Is.InRange(-10000.0, 10000.0));
                d = Utils.RandomDouble(0.0, 0.0);
                Assert.That(d, Is.InRange(0.0, 0.0));
            }
        }
    }
}
