namespace KS.NetTorch.Operations
{
    using MathNet.Numerics.LinearAlgebra;

    public sealed class AccumulateGradientFunction : IGradientFunction
    {
        private Tensor _tensor;

        public AccumulateGradientFunction(Tensor tensor)
        {
            _tensor = tensor;
        }

        public void ExecuteBackward(Matrix<double> initialGradient)
        {
            _tensor.AccumulateGradient_(initialGradient);
        }
    }
}
