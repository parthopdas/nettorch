namespace KS.NetTorch
{
    using System.Collections.Generic;

    public sealed class Context
    {
        public IReadOnlyList<Tensor> SavedTensors { get; private set; }

        public Context SaveForBackward(params Tensor[] tensors)
        {
            SavedTensors = tensors;

            return this;
        }
    }
}
