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

        public IGradientFunction GradientFunction { get; private set; }

        public bool IsLeaf { get; private set; }


        public bool TracksGradient { get; private set; }

        public Tensor(double data, bool tracksGradient = false)
            : this(new[] { data }, tracksGradient)
        {
        }

        public Tensor(double[] data, bool tracksGradient = false)
            : this(Matrix<double>.Build.DenseOfColumnArrays(data), tracksGradient)
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

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            var tracksGradient = t1.TracksGradient || t2.TracksGradient;

            var gradientFunction = tracksGradient
                ? (IGradientFunction)new MultiplyOperation(new Context().SaveForBackward(t1, t2))
                : new NullGradientFunction();

            return new Tensor(MultiplyOperation.ExecuteForward(t1.Data, t2.Data))
            {
                IsLeaf = !tracksGradient,
                TracksGradient = tracksGradient,
                GradientFunction = gradientFunction,
            };
        }

        public void Backward(double initialGradient = 1.0)
        {
            GradientFunction.ExecuteBackward(Matrix<double>.Build.Dense(1, 1, initialGradient));
        }
    }
}
