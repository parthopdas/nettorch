namespace KS.NetTorch
{
    using System;
    using KS.NetTorch.Operations;
    using MathNet.Numerics.LinearAlgebra;

    public static class TensorExtensions
    {
        public static Tensor PointwiseAdd(this Tensor @this, Tensor t2)
        {
            var tracksGradient = @this.TracksGradient || t2.TracksGradient;

            var gradientFunction = tracksGradient
                ? (IGradientFunction)new PointwiseAddOperation(new Context().SaveForBackward(@this, t2))
                : new NullGradientFunction();

            return new Tensor(PointwiseAddOperation.ExecuteForward(@this.Data, t2.Data))
            {
                IsLeaf = !tracksGradient,
                TracksGradient = tracksGradient,
                GradientFunction = gradientFunction,
            };
        }

        public static Tensor PointwiseSubtract(this Tensor @this, Tensor t2)
        {
            return @this + (-1 * t2);
        }

        public static Tensor PointwiseMultiply(this Tensor @this, Tensor t2)
        {
            var tracksGradient = @this.TracksGradient || t2.TracksGradient;

            var gradientFunction = tracksGradient
                ? (IGradientFunction)new PointwiseMultiplyOperation(new Context().SaveForBackward(@this, t2))
                : new NullGradientFunction();

            return new Tensor(PointwiseMultiplyOperation.ExecuteForward(@this.Data, t2.Data))
            {
                IsLeaf = !tracksGradient,
                TracksGradient = tracksGradient,
                GradientFunction = gradientFunction,
            };
        }

        public static Tensor PointwisePower(this Tensor @this, double exponent)
        {
            var tracksGradient = @this.TracksGradient;

            var gradientFunction = tracksGradient
                ? (IGradientFunction)new PointwisePowerOperation(new Context().SaveForBackward(@this))
                : new NullGradientFunction();

            return new Tensor(PointwisePowerOperation.ExecuteForward(@this.Data, exponent))
            {
                IsLeaf = !tracksGradient,
                TracksGradient = tracksGradient,
                GradientFunction = gradientFunction,
            };
        }

        public static Tensor Mean(this Tensor @this)
        {
            var c = new Context();
            if (@this.TracksGradient)
            {
                c.SaveForBackward(@this);
            }
            else
            {
                c.SaveForBackward();
            }

            var op = new MeanOperation(new Context().SaveForBackward(@this));

            var gradientFunction = @this.TracksGradient ? (IGradientFunction)op
                : new NullGradientFunction();

            return new Tensor(op.ExecuteForward(@this.Data))
            {
                IsLeaf = !@this.TracksGradient,
                TracksGradient = @this.TracksGradient,
                GradientFunction = gradientFunction,
            };
        }
    }
}
