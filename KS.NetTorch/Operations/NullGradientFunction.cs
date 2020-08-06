namespace KS.NetTorch.Operations
{
    using MathNet.Numerics.LinearAlgebra;

    public sealed class NullGradientFunction : IGradientFunction
    {
        public void ExecuteBackward(Matrix<double> _)
        {
        }
    }
}
