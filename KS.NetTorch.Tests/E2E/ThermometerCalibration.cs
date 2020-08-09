namespace KS.NetTorch.Tests.E2E
{
    using System.Collections.Generic;
    using FluentAssertions;
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;
    using Xunit;

    public class ThermometerCalibration
    {
        [Fact]
        public void FitThermometerInitial()
        {
            var tCelcius = new Tensor(new[] { 0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0 });
            var tUnknown = new Tensor(new[] { 35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4 });
            var tUnknownNormalized = 0.1 * tUnknown;

            var w = new Tensor(new[] { 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, }, tracksGradient: true);
            var b = new Tensor(new[] { 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, }, tracksGradient: true);

            var losses = new List<double>();
            TrainingLoop(
                epochs: 5000,
                lr: 1e-2,
                w: w,
                b: b,
                t_u: tUnknownNormalized,
                t_c: tCelcius,
                losses);

            w.Data.AlmostEqual(Matrix<double>.Build.Dense(11, 1, 5.367089798), LibConstants.Epsilon).Should().BeTrue();
            b.Data.AlmostEqual(Matrix<double>.Build.Dense(11, 1, -17.301222521), LibConstants.Epsilon).Should().BeTrue();

            var expectedLosses = new[] { 80.364345454545457, 7.8601242827569671, 3.8285420219932123, 3.0921905112819972, 2.9576990108595358, 2.9331347050588819, 2.9286481381191631, 2.9278286855741626, 2.9276790160200168, 2.9276516795066656, 2.9276466866076962, }.ToMatrix();
            losses.ToArray().ToMatrix().AlmostEqual(expectedLosses, LibConstants.Epsilon).Should().BeTrue();
        }

        private Tensor CreateModel(Tensor tUnknown, Tensor w, Tensor b)
        {
            return w * tUnknown + b;
        }

        private Tensor DefineLossFunction(Tensor tPredicted, Tensor tCelcius)
        {
            var squared_diffs = (tPredicted - tCelcius).PointwisePower(2);
            return squared_diffs.Mean();
        }

        private (Tensor, Tensor) TrainingLoop(int epochs, double lr, Tensor w, Tensor b, Tensor t_u, Tensor t_c, IList<double> losses)
        {
            for (var epoch = 1; epoch < epochs + 1; epoch++)
            {
                w.ResetGradientToZero_();
                b.ResetGradientToZero_();

                var t_p = CreateModel(t_u, w, b);
                var loss = DefineLossFunction(t_p, t_c);
                loss.Backward(Matrix<double>.Build.Dense(11, 1, 1));

                var wgrad = w.Gradient.Sum();
                w.ModifyData_(d => d - (lr * wgrad));
                var bgrad = b.Gradient.Sum();
                b.ModifyData_(d => d - (lr * bgrad));

                if (epoch == 1 || epoch % 500 == 0)
                {
                    losses.Add(loss.Data[0,0]);
                }
            }

            return (w, b);
        }
    }
}
