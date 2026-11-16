# Synthetic original-model reference

`original_model_reference.npz` contains a small model state dictionary, synthetic
input, and expected CPU output produced by the author-provided `Swin_Unet.py`.
It contains no trained One2N weights or atmospheric observations. Arrays are
stored in NumPy format and loaded with `allow_pickle=False`.

The reference uses PyTorch 2.6.0, NumPy 1.26.4, and timm 1.0.15 on CPU, with
random seed 31, evaluation mode, and one PyTorch thread. The original model's
unconditional CUDA synchronization call was removed to permit CPU execution;
its tensor computations were unchanged.

Original source SHA-256:
`e512d838a0fea7fbe532a528b68d0d16f4172bf1a543f89274a6279ca29284ca`.

Architecture: input `(1, 5, 5, 16, 16)`, patch `(1, 2, 2)`, embedding 8,
depths `(2, 4, 4, 2)`, heads `(2, 2, 2, 2)`, window `(5, 2, 2)`, and absolute
position embedding and patch normalization enabled. State arrays use the
`state::` prefix; `inputs` and `expected` provide the forward regression case.

`test_model.py` loads the frozen state strictly and compares the result to
the original output with `rtol=2e-6, atol=2e-6` to allow small CPU kernel
rounding differences. This validates a small model computation, not a trained
global forecast or prediction skill.
