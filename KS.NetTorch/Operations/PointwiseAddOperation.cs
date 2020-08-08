namespace KS.NetTorch.Operations
{
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra;

    public class PointwiseAddOperation : IGradientFunction
    {
        private readonly Context _context;

        private readonly (IGradientFunction GF, int Int)[] _nextFunctions;

        public PointwiseAddOperation(Context context)
        {
            _context = context;
            _nextFunctions = context
                .SavedTensors
                .Select(t => (Function: t.GradientFunction, Int: 0))
                .ToArray();
        }

        public static Matrix<double> ExecuteForward(params Matrix<double>[] args)
        {
            return args[0] + args[1];
        }

        public void ExecuteBackward(Matrix<double> initialGradient)
        {
            _nextFunctions[0].GF.ExecuteBackward(initialGradient);
            _nextFunctions[1].GF.ExecuteBackward(initialGradient);
        }
    }
}
