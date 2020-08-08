namespace KS.NetTorch.Operations
{
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra;

    public sealed class MeanOperation : IGradientFunction
    {
        private int _count;
        private readonly Context _context;

        private readonly (IGradientFunction GF, int Int)[] _nextFunctions;

        public MeanOperation(Context context)
        {
            _context = context;
            _nextFunctions = context
                .SavedTensors
                .Select(t => (Function: t.GradientFunction, Int: 0))
                .ToArray();
        }

        public Matrix<double> ExecuteForward(Matrix<double> m)
        {
            _count = (m.RowCount * m.ColumnCount);
            return Matrix<double>.Build.Dense(1, 1, m.ColumnSums().Sum() / _count);
        }

        public void ExecuteBackward(Matrix<double> initialGradient)
        {
            _nextFunctions[0].GF.ExecuteBackward(initialGradient / _count);
        }
    }
}
