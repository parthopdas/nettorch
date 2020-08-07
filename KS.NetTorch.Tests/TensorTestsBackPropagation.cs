namespace KS.NetTorch.Tests
{
    using FluentAssertions;
    using KS.NetTorch.Operations;
    using KS.NetTorch.Tests.Extensions;
    using MathNet.Numerics.LinearAlgebra;
    using Xunit;

    public class TensorTestsBackPropagation
    {
        private static readonly Matrix<double> Zero1x1 = 0.0.ToMatrix();
        private static readonly Matrix<double> Zero2x1 = new[] { 0d, 0 }.ToMatrix();

        [Fact]
        public void BackPropagateOnScalarMultiplyOneOfThemTracksGradient()
        {
            var a = new Tensor(2, tracksGradient: true);
            var b = new Tensor(3);

            var c = a * b;

            c.Backward(2.0);


            a.Should().HaveProperties(2.ToMatrix(), 6.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            b.Should().HaveProperties(3.ToMatrix(), Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false);
        }

        [Fact]
        public void BackPropagateOnTensorMultiplyOneOfThemTracksGradient()
        {
            var a = new Tensor(new[] { 2d, 4 }, tracksGradient: true);
            var b = new Tensor(new[] { 7d, 11 });

            var c = a * b;

            c.Backward(new[] { 1d, 2.0 }.ToMatrix());


            a.Should().HaveProperties(new[] { 2d, 4 }.ToMatrix(), new[] { 7d, 22 }.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            b.Should().HaveProperties(new[] { 7d, 11 }.ToMatrix(), Zero2x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false);
        }

        [Fact]
        public void BackPropagateOn3WayMultipleWithAllTrackingGradiends()
        {
            var a = new Tensor(2, tracksGradient: true);
            var b = new Tensor(3, tracksGradient: true);

            var c = a * b;

            var d = new Tensor(4, tracksGradient: true);

            var e = c * d;

            c.Backward(1.0);
            e.Backward();

            e.Should().HaveProperties(24.ToMatrix(), Zero1x1, typeof(PointwiseMultiplyOperation), isLeaf: false, tracksGradient: true);
            c.Should().HaveProperties(6.ToMatrix(), Zero1x1, typeof(PointwiseMultiplyOperation), isLeaf: false, tracksGradient: true);
            d.Should().HaveProperties(4.ToMatrix(), 6.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            b.Should().HaveProperties(3.ToMatrix(), 10.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            a.Should().HaveProperties(2.ToMatrix(), 15.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
        }
    }
}
