namespace KS.NetTorch
{
    using System;
    using MathNet.Numerics.LinearAlgebra;

    public static class MatrixExtensions
    {

        public static Matrix<double> ToMatrix(this int i)
        {
            return ((double)i).ToMatrix();
        }
        public static Matrix<double> ToMatrix(this double d)
        {
            return new[] { d }.ToMatrix();
        }

        public static Matrix<double> ToMatrix(this double[] v)
        {
            return Matrix<double>.Build.DenseOfColumnArrays(v);
        }

        public static Matrix<double> ToMatrix(this double[,] m)
        {
            return Matrix<double>.Build.DenseOfArray(m);
        }
    }
}
