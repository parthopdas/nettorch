namespace KS.NetTorch
{
    using System.Diagnostics;
    using KS.NetTorch.Operations;
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

    [DebuggerDisplay("Rank 2: {Data.RowCount}x{Data.ColumnCount}")]
    public sealed class Tensor
    {
        public Matrix<double> Data { get; private set; }

        public Matrix<double> Gradient { get; private set; }

        public IGradientFunction GradientFunction { get; private set; }

        public bool IsLeaf { get; private set; }


        public bool RequiresGradient { get; private set; }

        public Tensor(double data, bool requiresGradient = false)
            : this(new[] { data }, requiresGradient)
        {
        }

        public Tensor(double[] data, bool requiresGradient = false)
            : this(Matrix<double>.Build.DenseOfColumnArrays(data), requiresGradient)
        {
        }

        public Tensor(Matrix<double> data, bool requiresGradient = false)
        {
            Data = data;
            Gradient = Matrix<double>.Build.Dense(data.RowCount, data.ColumnCount, 0);
            GradientFunction = requiresGradient
                ? new AccumulateGradientFunction(this) as IGradientFunction
                : new NullGradientFunction();
            IsLeaf = true;
            RequiresGradient = requiresGradient;
        }

        public void AccumulateGradient_(Matrix<double> delta)
        {
            Gradient = Gradient.Add(delta);
        }

        public override string ToString() => Data.ToString();

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            var requiresGradient = t1.RequiresGradient || t2.RequiresGradient;

            var gradientFunction = requiresGradient
                ? (IGradientFunction)new MultiplyBackwardGradientFunction(new Context().SaveForBackward(t1, t2))
                : new NullGradientFunction();

            return new Tensor(t1.Data * t2.Data)
            {
                IsLeaf = !requiresGradient,
                RequiresGradient = requiresGradient,
                GradientFunction = gradientFunction,
            };
        }

        public void Backward(double initialGradient = 1.0)
        {
            GradientFunction.Execute(Matrix<double>.Build.Dense(1, 1, initialGradient));
        }
    }
}
