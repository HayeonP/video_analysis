# Vendored TransNetV2 inference model

`model.py` is the unmodified `inference-pytorch/transnetv2_pytorch.py` from:
https://github.com/soCzech/TransNetV2/tree/85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed

The upstream MIT license is included in `LICENSE`.

Converted PyTorch weights are downloaded from the community mirror:
https://huggingface.co/magnusdtd/TransNetV2/blob/1c0e86a1a6333ff57c4458a44fe4b5b4a0faae2a/transnetv2-pytorch-weights.pth

SHA256: `eed5336d5d6a013c67f5863505a26e7e835053e64a9ce413d6b089ccba07bb53`

`gradual.py` verifies this checksum and loads only tensors with `weights_only=True`.
The inference implementation uses the original 48×27 RGB inputs, 100-frame windows,
and 50-frame overlap, with edge padding. PySceneDetect supplies frame timecodes and
FFmpeg export; its Adaptive/Content detectors remain available for comparison.
