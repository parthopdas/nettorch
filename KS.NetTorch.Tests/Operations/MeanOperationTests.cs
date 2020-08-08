namespace KS.NetTorch.Tests.Operations
{
    using FluentAssertions;
    using KS.NetTorch.Operations;
    using KS.NetTorch.Tests.Extensions;
    using MathNet.Numerics.LinearAlgebra;
    using Xunit;

    public sealed class MeanOperationTests
    {
        private static readonly Matrix<double> Zero1x1 = 0.0.ToMatrix();

        [Fact]
        public void TestMeanOfTensorWithoutGradientTracking()
        {
            var t = new Tensor(new[] { 0.5, 14.0, -15.0, });

            var m = t.Mean();

            m.Backward(new[] { 1d, 1, 1, }.ToMatrix());

            m.Should().HaveProperties(
                -0.1666666716337204.ToMatrix(),
                Zero1x1,
                typeof(NullGradientFunction),
                isLeaf: true,
                tracksGradient: false);

            t.Should().HaveProperties(
                new[] { 0.5, 14.0, -15.0 }.ToMatrix(),
                new[] { 0d, 0, 0 }.ToMatrix(),
                typeof(NullGradientFunction),
                isLeaf: true,
                tracksGradient: false);
        }

        [Fact]
        public void TestMeanOfTensorWithGradientTracking()
        {
            var t = new Tensor(new[] { 0.5, 14.0, -15.0, }, tracksGradient: true);

            var m = t.Mean();

            m.Backward(new[] { 1d, 1, 1, }.ToMatrix());

            m.Should().HaveProperties(
                -0.1666666716337204.ToMatrix(),
                Zero1x1,
                typeof(MeanOperation),
                isLeaf: false,
                tracksGradient: true);

            t.Should().HaveProperties(
                new[] { 0.5, 14.0, -15.0 }.ToMatrix(),
                new[] { 1d / 3, 1d / 3, 1d / 3 }.ToMatrix(),
                typeof(AccumulateGradientFunction),
                isLeaf: true,
                tracksGradient: true);
        }
    }
}
