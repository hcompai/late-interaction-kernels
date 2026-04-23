{
  description = "Flake for late-interaction-kernels (Triton MaxSim / PLAID / fused-head)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;
      # Triton kernels auto-tune on first call; skip import-time check.
      doGetKernelCheck = false;
    };
}
