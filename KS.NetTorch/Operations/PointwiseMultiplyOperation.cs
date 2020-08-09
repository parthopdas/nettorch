namespace KS.NetTorch.Operations
{
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra;

    public sealed class PointwiseMultiplyOperation : IGradientFunction
    {
        private readonly Context _context;

        private readonly (IGradientFunction GF, int Int)[] _nextFunctions;

        public PointwiseMultiplyOperation(Context context)
        {
            _context = context;
            _nextFunctions = context
                .SavedTensors
                .Select(t => (Function: t.GradientFunction, Int: 0))
                .ToArray();
        }

        public static Matrix<double> ExecuteForward(Matrix<double> m1, Matrix<double> m2)
        {
            var (m1b, m2b) = LibMatrixExtensions.Broadcast(m1, m2);
            return m1b.PointwiseMultiply(m2b);
        }

        public void ExecuteBackward(Matrix<double> initialGradient)
        {
            _nextFunctions[0].GF.ExecuteBackward(_context.SavedTensors[1].Data.PointwiseMultiply(initialGradient));
            _nextFunctions[1].GF.ExecuteBackward(_context.SavedTensors[0].Data.PointwiseMultiply(initialGradient));
        }
    }
}
