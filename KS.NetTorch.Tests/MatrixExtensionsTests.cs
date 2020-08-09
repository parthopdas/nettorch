namespace KS.NetTorch.Tests
{
    using FluentAssertions;
    using KS.NetTorch.Tests.Extensions;
    using MathNet.Numerics;
    using Xunit;

    public sealed class MatrixExtensionsTests
    {
        [Fact]
        public void BroadcastLHSTests()
        {
            var m1 = new[] { 1d, }.ToMatrix();
            var m2 = new[] { 1d, 2d }.ToMatrix();

            var (m1b, m2b) = LibMatrixExtensions.Broadcast(m1, m2);

            m1b.AlmostEqual(m1, LibConstants.Epsilon).Should().BeTrue();
            m2b.AlmostEqual(m2, LibConstants.Epsilon).Should().BeTrue();
        }
    }
}
