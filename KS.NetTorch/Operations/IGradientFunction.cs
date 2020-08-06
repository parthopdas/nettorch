namespace KS.NetTorch.Operations
{
    using MathNet.Numerics.LinearAlgebra;

    public interface IGradientFunction
    {
        void ExecuteBackward(Matrix<double> initialGradient);
    }
}
