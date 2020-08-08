namespace KS.NetTorch
{
    using System.Diagnostics;
    using KS.NetTorch.Operations;
    using MathNet.Numerics.LinearAlgebra;

    [DebuggerDisplay("Rank 2: {Data.RowCount}x{Data.ColumnCount}")]
    public sealed class Tensor
    {
        public Matrix<double> Data { get; private set; }

        public Matrix<double> Gradient { get; private set; }

        // TODO: private setter
        public IGradientFunction GradientFunction { get; set; }

        // TODO: private setter
        public bool IsLeaf { get; set; }

        // TODO: private setter
        public bool TracksGradient { get; set; }

        public Tensor(double data, bool tracksGradient = false)
            : this(data.ToMatrix(), tracksGradient)
        {
        }

        public Tensor(double[] data, bool tracksGradient = false)
            : this(data.ToMatrix(), tracksGradient)
        {
        }

        public Tensor(double[,] data, bool tracksGradient = false)
            : this(data.ToMatrix(), tracksGradient)
        {
        }

        public Tensor(Matrix<double> data, bool tracksGradient = false)
        {
            Data = data;
            Gradient = Matrix<double>.Build.Dense(data.RowCount, data.ColumnCount, 0);
            GradientFunction = tracksGradient
                ? new AccumulateGradientFunction(this) as IGradientFunction
                : new NullGradientFunction();
            IsLeaf = true;
            TracksGradient = tracksGradient;
        }

        public void AccumulateGradient_(Matrix<double> delta)
        {
            Gradient = Gradient.Add(delta);
        }

        public override string ToString() => Data.ToString();

        public void Backward(double initialGradient = 1.0)
        {
            Backward(Matrix<double>.Build.Dense(1, 1, initialGradient));
        }

        public void Backward(Matrix<double> initialGradient)
        {
            GradientFunction.ExecuteBackward(initialGradient);
        }

        public static Tensor operator *(double d, Tensor t1)
        {
            return t1 * d;
        }

        public static Tensor operator *(Tensor t1, double d)
        {
            var t2 = new Tensor(Matrix<double>.Build.Dense(t1.Data.RowCount, t1.Data.ColumnCount, d));

            return t1 * t2;
        }

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            return t1.PointwiseMultiply(t2);
        }

        public static Tensor operator +(Tensor t1, Tensor t2)
        {
            return t1.PointwiseAdd(t2);
        }

        public static Tensor operator -(Tensor t1, Tensor t2)
        {
            return t1.PointwiseSubtract(t2);
        }
    }
}
