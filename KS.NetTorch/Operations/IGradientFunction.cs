namespace KS.NetTorch.Operations
{
    using MathNet.Numerics.LinearAlgebra;

    public interface IGradientFunction
    {
        void Execute(Matrix<double> initialGradient);
    }
}
