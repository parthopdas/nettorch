namespace KS.NetTorch.Tests.E2E
{
    using MathNet.Numerics.LinearAlgebra;
    using Xunit;

    public class ThermometerCalibration
    {
        private Tensor CreateModel(Tensor tUnknown, Tensor w, Tensor b)
        {
            return w * tUnknown + b;
        }

        private Tensor DefineLossFunction(Tensor tPredicted, Tensor tCelcius)
        {
            var squared_diffs = (tPredicted - tCelcius).PointwisePower(2);
            return squared_diffs.Mean();
        }

        //[Fact]
        public void FitThermometerInitial()
        {
            var tCelcius = new Tensor(new[] { 0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0 });
            var tUnknown = new Tensor(new[] { 35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4 });
            var tUnknownNormalized = 0.1 * tUnknown;

            //var w = new Tensor(new[] { 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, }, tracksGradient: true);
            //var b = new Tensor(new[] { 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, }, tracksGradient: true);
            var w = new Tensor(new[] { 1d, }, tracksGradient: true);
            var b = new Tensor(new[] { 0d, }, tracksGradient: true);

            var m = CreateModel(tUnknown, w, b);
            m.Backward(Matrix<double>.Build.Dense(11, 1, 1));

            var loss = DefineLossFunction(m, tCelcius);
            loss.Backward();

        }
    }
}
