namespace KS.NetTorch.Tests.E2E
{
    using Xunit;

    public class ThermometerCalibration
    {
        //[Fact]
        public void FitThermometer()
        {
            var t_c = new Tensor(new[] { 0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0 });
            var t_u = new Tensor(new[] { 35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4 });
            var t_un = 0.1 * t_u;

        }
    }
}
