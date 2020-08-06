namespace KS.NetTorch.Operations
{
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra;

    public sealed class MultiplyBackwardGradientFunction : IGradientFunction
    {
        private readonly Context _context;

        private readonly (IGradientFunction GF, int Int)[] _nextFunctions;

        public MultiplyBackwardGradientFunction(Context context)
        {
            _context = context;
            _nextFunctions = context
                .SavedTensors
                .Select(t => (Function: t.GradientFunction, Int: 0))
                .ToArray();
        }

        public void Execute(Matrix<double> initialGradient)
        {
            _nextFunctions[0].GF.Execute(_context.SavedTensors[1].Data.PointwiseMultiply(initialGradient));
            _nextFunctions[1].GF.Execute(_context.SavedTensors[0].Data.PointwiseMultiply(initialGradient));
        }
    }
}
