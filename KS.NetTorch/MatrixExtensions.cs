namespace KS.NetTorch
{
    using MathNet.Numerics.LinearAlgebra;

    public static class MatrixExtensions
    {
        public static Matrix<double> ToMatrix(this double d)
        {
            return Matrix<double>.Build.Dense(1, 1, d);
        }

        public static Matrix<double> ToMatrix(this int i)
        {
            return Matrix<double>.Build.Dense(1, 1, i);
        }
    }
}
