using System;
using NUnit.Framework;

namespace Visionary.ConvolutionalNeuralNetworkTest
{
    using Visionary.ConvolutionalNeuralNetwork;

    [TestFixture]
    internal class UtilsTest
    {
        [Test]
        public void SIGMOIDTest()
        {
            double x = 9.9;
            Assert.That(1.7159 * Math.Tanh(0.66666667 * x), Is.EqualTo(Utils.SIGMOID(x)));
            x = -3.9;
            Assert.That(1.7159 * Math.Tanh(0.66666667 * x), Is.EqualTo(Utils.SIGMOID(x)));
            x = 0;
            Assert.That(1.7159 * Math.Tanh(0.66666667 * x), Is.EqualTo(Utils.SIGMOID(x)));
            x = 9.0;
            Assert.That(2 / (1 + Math.Exp(-2 * x)) - 1, Is.EqualTo(Utils.SIGMOID(x)));
        }

        [Test]
        public void DSIGMOIDTest()
        {
            double x = 9.9;
            Assert.That(
                0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x), Is.EqualTo(Utils.DSIGMOID(x)));
            x = -3.9;
            Assert.That(
                0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x), Is.EqualTo(Utils.DSIGMOID(x)));
            x = 0.0;
            Assert.That(
                0.66666667 / 1.7159 * (1.7159 + x) * (1.7159 - x), Is.EqualTo(Utils.DSIGMOID(x)));
        }
    }
}
