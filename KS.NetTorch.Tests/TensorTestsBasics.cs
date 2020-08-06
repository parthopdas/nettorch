namespace KS.NetTorch.Tests
{
    using FluentAssertions;
    using KS.NetTorch.Operations;
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;
    using Xunit;

    public class TensorTestsBasics
    {
        private static readonly Matrix<double> Zero1x1 = 0.0.ToMatrix();

        [Fact]
        public void CreateATensor()
        {
            var a = new Tensor(2);

            a.Gradient.AlmostEqual(Zero1x1, LibConstants.Epsilon).Should().BeTrue();
            a.GradientFunction.Should().BeOfType<NullGradientFunction>();
            a.IsLeaf.Should().BeTrue();
            a.RequiresGradient.Should().BeFalse();
            a.ToString().Should().Be(@"DenseMatrix 1x1-Double
2
");
        }

        [Fact]
        public void CreateATensorWithGradientTracking()
        {
            var a = new Tensor(2, requiresGradient: true);

            a.Data.AlmostEqual(2.ToMatrix(), LibConstants.Epsilon).Should().BeTrue();
            a.Gradient.AlmostEqual(Zero1x1, LibConstants.Epsilon).Should().BeTrue();
            a.GradientFunction.Should().BeOfType<AccumulateGradientFunction>();
            a.IsLeaf.Should().BeTrue();
            a.RequiresGradient.Should().BeTrue();
        }

        [Fact]
        public void Multiply2UserCreatedTensorsThatDontTrackGradient()
        {
            var a = new Tensor(2);
            var b = new Tensor(3);

            var c = a * b;

            c.Data.AlmostEqual(6.ToMatrix(), LibConstants.Epsilon).Should().BeTrue();
            c.Gradient.AlmostEqual(Zero1x1, LibConstants.Epsilon).Should().BeTrue();
            c.GradientFunction.Should().BeOfType<NullGradientFunction>();
            c.IsLeaf.Should().BeTrue();
            c.RequiresGradient.Should().BeFalse();
        }

        [Fact]
        public void Multiply2UserCreatedTensorsOneOfThemTracksGradient()
        {
            var a = new Tensor(2, requiresGradient: true);
            var b = new Tensor(3);

            var c = a * b;

            c.Data.AlmostEqual(6.ToMatrix(), LibConstants.Epsilon).Should().BeTrue();
            c.Gradient.AlmostEqual(Zero1x1, LibConstants.Epsilon).Should().BeTrue();
            c.GradientFunction.Should().BeOfType<MultiplyBackwardGradientFunction>();
            c.IsLeaf.Should().BeFalse();
            c.RequiresGradient.Should().BeTrue();
        }
    }
}
