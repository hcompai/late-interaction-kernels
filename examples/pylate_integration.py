"""Drop flash-colbert into an existing PyLate training loop.

Result: training step runs the Triton MaxSim instead of PyTorch's einsum+max.
Everything else in PyLate is untouched.
"""

from flash_colbert.pylate_compat import patch_pylate

patch_pylate()

# ...now use PyLate exactly as you normally would...
#
# from pylate import models, losses
# model = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1")
# loss = losses.Contrastive(model=model)
# trainer = ...
# trainer.train()
