namespace KS.NetTorch.Operations
{
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra;

    public sealed class PointwisePowerOperation : IGradientFunction
    {
        private readonly Context _context;

        private readonly (IGradientFunction GF, int Int)[] _nextFunctions;

        public PointwisePowerOperation(Context context)
        {
            _context = context;
            _nextFunctions = context
                .SavedTensors
                .Select(t => (Function: t.GradientFunction, Int: 0))
                .ToArray();
        }

        public static Matrix<double> ExecuteForward(Matrix<double> m, double exponent)
        {
            return m.PointwisePower(exponent);
        }

        public void ExecuteBackward(Matrix<double> initialGradient)
        {
            // TODO: Generalize this to exponent.
            _nextFunctions[0].GF.ExecuteBackward(_context.SavedTensors[0].Data.Multiply(2).PointwiseMultiply(initialGradient));
        }
    }
}
