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

        public static Matrix<double> ExecuteForward(params Matrix<double>[] args)
        {
            return args[0].PointwiseMultiply(args[1]);
        }

        public void ExecuteBackward(Matrix<double> initialGradient)
        {
            _nextFunctions[0].GF.ExecuteBackward(_context.SavedTensors[1].Data.PointwiseMultiply(initialGradient));
            _nextFunctions[1].GF.ExecuteBackward(_context.SavedTensors[0].Data.PointwiseMultiply(initialGradient));
        }
    }
}
