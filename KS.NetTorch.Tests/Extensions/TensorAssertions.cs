namespace KS.NetTorch.Tests.Extensions
{
    using System;
    using FluentAssertions;
    using FluentAssertions.Execution;
    using FluentAssertions.Primitives;
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;

    public class TensorAssertions : ReferenceTypeAssertions<Tensor, TensorAssertions>
    {
        public TensorAssertions(Tensor instance)
        {
            Subject = instance;
        }

        protected override string Identifier => nameof(Tensor).ToLowerInvariant();

        public AndConstraint<TensorAssertions> HaveProperties(
            Matrix<double> data, Matrix<double> gradient, Type gradientFunctionType, bool isLeaf, bool tracksGradient, string because = "", params object[] becauseArgs)
        {
            Execute.Assertion
                .BecauseOf(because, becauseArgs)
                .ForCondition(data.AlmostEqual(Subject.Data, LibConstants.Epsilon))
                .FailWith("Data mismatch: Expected: {0}; Given: {1}", data, Subject.Data)
                .Then
                .ForCondition(gradient.AlmostEqual(Subject.Gradient, LibConstants.Epsilon))
                .FailWith("Gradient mismatch: Expected: {0}; Given: {1}", gradient, Subject.Gradient)
                .Then
                .ForCondition(gradientFunctionType.Equals(Subject.GradientFunction.GetType()))
                .FailWith("Gradient function mismatch: Expected: {0}; Given: {1}", gradientFunctionType, Subject.GradientFunction.GetType())
                .Then
                .ForCondition(isLeaf == Subject.IsLeaf)
                .FailWith("IsLeaf mismatch: Expected: {0}; Given: {1}", isLeaf, Subject.IsLeaf)
                .Then
                .ForCondition(tracksGradient == Subject.TracksGradient)
                .FailWith("GradientRequired mismatch: Expected: {0}; Given: {1}", tracksGradient, Subject.TracksGradient)
                ;

            return new AndConstraint<TensorAssertions>(this);
        }
    }
}
