namespace KS.NetTorch.Tests
{
    using FluentAssertions;
    using KS.NetTorch.Operations;
    using KS.NetTorch.Tests.Extensions;
    using MathNet.Numerics.LinearAlgebra;
    using Xunit;

    public sealed class TensorTestsBasics
    {
        private static readonly Matrix<double> Zero1x1 = 0.0.ToMatrix();

        [Fact]
        public void CreateATensor()
        {
            var a = new Tensor(2);

            a.Should().HaveProperties(2.ToMatrix(), Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false);
            a.ToString().Should().Be(@"DenseMatrix 1x1-Double
2
");
        }

        [Fact]
        public void CreateATensorWithGradientTracking()
        {
            var a = new Tensor(2, tracksGradient: true);

            a.Should().HaveProperties(2.ToMatrix(), Zero1x1, typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true);
            a.ToString().Should().Be(@"DenseMatrix 1x1-Double
2
");
        }
    }
}
