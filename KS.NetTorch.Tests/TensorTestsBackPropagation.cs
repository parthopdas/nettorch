namespace KS.NetTorch.Tests
{
    using FluentAssertions;
    using KS.NetTorch.Operations;
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;
    using Xunit;

    public class TensorTestsBackPropagation
    {
        private static readonly Matrix<double> Zero1x1 = 0.0.ToMatrix();

        [Fact]
        public void BackPropagateOnMultiply2UserCreatedTensorsOneOfThemTracksGradient()
        {
            var a = new Tensor(2, requiresGradient: true);
            var b = new Tensor(3);

            var c = a * b;

            c.Backward(2.0);

            a.Data.AlmostEqual(2.ToMatrix(), LibConstants.Epsilon).Should().BeTrue();
            a.Gradient.AlmostEqual(6.ToMatrix(), LibConstants.Epsilon).Should().BeTrue();
            a.GradientFunction.Should().BeOfType<AccumulateGradientFunction>();
            a.IsLeaf.Should().BeTrue();
            a.RequiresGradient.Should().BeTrue();

            b.Data.AlmostEqual(3.ToMatrix(), LibConstants.Epsilon).Should().BeTrue();
            b.Gradient.AlmostEqual(Zero1x1, LibConstants.Epsilon).Should().BeTrue();
            b.GradientFunction.Should().BeOfType<NullGradientFunction>();
            b.IsLeaf.Should().BeTrue();
            b.RequiresGradient.Should().BeFalse();
        }
    }
}
