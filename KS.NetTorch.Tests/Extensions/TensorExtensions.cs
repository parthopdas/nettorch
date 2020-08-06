namespace KS.NetTorch.Tests.Extensions
{
    public static class TensorExtensions
    {
        public static TensorAssertions Should(this Tensor instance)
        {
            return new TensorAssertions(instance);
        }
    }
}
