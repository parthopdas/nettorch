namespace KS.NetTorch.Tests.Operations
{
    using System.Collections.Generic;
    using Extensions;
    using MathNet.Numerics.LinearAlgebra;
    using NetTorch.Operations;
    using Xunit;

    public sealed class PointwiseMultiplyOperationTests
    {
        private static readonly Matrix<double> Zero1x1 = 0.0.ToMatrix();

        public static IEnumerable<object[]> TestData =>
            new List<object[]>
            {
                // Test: Neither tracking gradients
                new object[]
                {
                    (2d.ToMatrix(), false),
                    (3d.ToMatrix(), false),
                    6d.ToMatrix(),
                    (Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false),
                    (Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false),
                    (Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false),
                },
                // Test: One tracking gradients
                new object[]
                {
                    (2.ToMatrix(), true),
                    (3.ToMatrix(), false),
                    6d.ToMatrix(),
                    (3.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true),
                    (Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false),
                    (Zero1x1, typeof(PointwiseMultiplyOperation), isLeaf: false, tracksGradient: true),
                },
                // Test: Both tracking gradients
                new object[]
                {
                    (2.ToMatrix(), true),
                    (3.ToMatrix(), true),
                    6d.ToMatrix(),
                    (3.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true),
                    (2.ToMatrix(), typeof(AccumulateGradientFunction), isLeaf: true, tracksGradient: true),
                    (Zero1x1, typeof(PointwiseMultiplyOperation), isLeaf: false, tracksGradient: true),
                },

                // Rhs boxing: Test with one gradient tracking
                // new object[]
                // {
                //     (new[] {1d, 2d, }.ToMatrix(), false),
                //     (new[] {3d, }.ToMatrix(), false),
                //     new[] {3d, 6d, }.ToMatrix(),
                //     (Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false),
                //     (Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false),
                //     (Zero1x1, typeof(NullGradientFunction), isLeaf: true, tracksGradient: false),
                // },
                // Rhs boxing: Test with both grad tracking
                // Rhs boxing: Test with neither grad tracking

                // Lhs boxing: Test with one gradient tracking
                // Lhs boxing: Test with both grad tracking
                // Lhs boxing: Test with neither grad tracking
            };

        [Theory]
        [MemberData(nameof(TestData))]
        public void TestPointwiseMultiplyOperation(dynamic taProps, dynamic tbProps, dynamic tcData, dynamic aProps, dynamic bProps, dynamic cProps)
        {
            var a = new Tensor(taProps.Item1 as Matrix<double>, taProps.Item2);
            var b = new Tensor(tbProps.Item1 as Matrix<double>, tbProps.Item2);
            var c = a * b;

            c.Backward();

            a.Should().HaveProperties(taProps.Item1, aProps.Item1, aProps.Item2, aProps.Item3, aProps.Item4);
            b.Should().HaveProperties(tbProps.Item1, bProps.Item1, bProps.Item2, bProps.Item3, bProps.Item4);
            c.Should().HaveProperties(tcData, cProps.Item1, cProps.Item2, cProps.Item3, cProps.Item4);
        }
    }
}
