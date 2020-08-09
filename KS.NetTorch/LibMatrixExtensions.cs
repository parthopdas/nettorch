namespace KS.NetTorch
{
    using MathNet.Numerics.LinearAlgebra;

    public static class LibMatrixExtensions
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

        public static (Matrix<double>, Matrix<double>) Broadcast(Matrix<double> m1, Matrix<double> m2)
        {
            return (m1, m2);
        }

        public static double Sum(this Matrix<double> @this)
        {
            return @this.ColumnSums().Sum();
        }
    }
}
