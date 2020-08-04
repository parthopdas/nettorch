namespace KS.NetTorch.Tests
{
    using FluentAssertions;
    using Xunit;

    public class TensorTests
    {
        [Fact]
        public void CreateATensor()
        {
            var x = new Tensor(new double[] { 1, 2, 3, 4, 5 });

            x.ToString().Should().Be(@"DenseVector 5-Double
1
2
3
4
5
");
        }

        [Fact]
        public void Add2Tensors()
        {
            var x = new Tensor(new double[] { 1, 2, 3, 4, 5 });

            var y = x + x;

            y.ToString().Should().Be(@"DenseVector 5-Double
 2
 4
 6
 8
10
");
        }
    }
}
