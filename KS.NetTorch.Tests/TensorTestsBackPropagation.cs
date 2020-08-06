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

        [Fact]
        public void BackPropagateOnMultiply2UserCreatedTensorsOneOfThemTracksGradient()
        {
            var a = new Tensor(2, tracksGradient: true);
            var b = new Tensor(3);

            var c = a * b;

            c.Backward(2.0);


            a.Should().HaveProperties(2.ToMatrix(), 6.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            b.Should().HaveProperties(3.ToMatrix(), Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false);
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

            e.Should().HaveProperties(24.ToMatrix(), Zero1x1, typeof(MultiplyOperation), isLeaf: false, tracksGradient: true);
            c.Should().HaveProperties(6.ToMatrix(), Zero1x1, typeof(MultiplyOperation), isLeaf: false, tracksGradient: true);
            d.Should().HaveProperties(4.ToMatrix(), 6.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            b.Should().HaveProperties(3.ToMatrix(), 10.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            a.Should().HaveProperties(2.ToMatrix(), 15.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
        }
    }
}
