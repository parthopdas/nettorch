namespace KS.NetTorch
{
    using System.Diagnostics;
    using MathNet.Numerics.LinearAlgebra;

    [DebuggerDisplay("Rank 2: {_data.RowCount}x{{_data.ColumnCount}}")]
    public sealed class Tensor
    {
        private readonly Vector<double> _data;

        public Tensor(double[] data)
            : this(Vector<double>.Build.DenseOfArray(data))
        {
        }

        private Tensor(Vector<double> data)
        {
            _data = data;
        }

        public override string ToString()
        {
            return _data.ToString();
        }
        public static Tensor operator +(Tensor t1, Tensor t2)
            => new Tensor(t1._data + t2._data);
    }
}
