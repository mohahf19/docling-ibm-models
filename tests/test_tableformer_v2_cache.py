# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from docling_ibm_models.tableformer_v2.model import (
    CachedTransformerDecoderLayer,
    TableFormerV2,
    TableFormerV2Config,
)


@pytest.mark.parametrize("mask_type", ["bool2d", "float3d"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("prefix", [1, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_cached_decode_matches_pytorch_decoder(batch, prefix, masked, mask_type):
    torch.manual_seed(7)
    layer = CachedTransformerDecoderLayer(16, 4, 32).eval()
    reference = nn.TransformerDecoderLayer(
        16, 4, 32, dropout=0, batch_first=True
    ).eval()
    reference.load_state_dict(layer.state_dict())
    inputs = torch.randn(batch, 9, 16)
    memory = torch.randn(batch, 7, 16)
    causal = torch.ones(9, 9, dtype=torch.bool).triu(1)
    padding = torch.zeros(batch, 9, dtype=torch.bool) if masked else None
    memory_padding = torch.zeros(batch, 7, dtype=torch.bool) if masked else None
    if masked:
        padding[:, 2] = True
        memory_padding[:, -1] = True
    if mask_type == "float3d":
        causal = torch.zeros(batch * 4, 9, 9).masked_fill(causal, float("-inf"))
        causal[1::2, :, 1] = float("-inf")
        if masked:
            padding = torch.zeros_like(padding, dtype=torch.float32).masked_fill(
                padding, float("-inf")
            )
            memory_padding = torch.zeros_like(
                memory_padding, dtype=torch.float32
            ).masked_fill(memory_padding, float("-inf"))
    with torch.no_grad():
        expected = reference(
            inputs,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
            memory_key_padding_mask=memory_padding,
        )
        output, cache = layer(
            inputs[:, :prefix],
            memory,
            tgt_mask=causal[..., :prefix, :prefix],
            tgt_key_padding_mask=padding[:, :prefix] if masked else None,
            memory_key_padding_mask=memory_padding,
            use_cache=True,
        )
        outputs = [output]
        cross_storage = [cache[i].data_ptr() for i in [2, 3]]
        for i in range(prefix, inputs.shape[1]):
            output, cache = layer(
                inputs[:, i : i + 1],
                memory,
                tgt_mask=causal[..., i : i + 1, : i + 1],
                tgt_key_padding_mask=padding[:, : i + 1] if masked else None,
                memory_key_padding_mask=memory_padding,
                past_key_value=cache,
                use_cache=True,
            )
            assert cache[0].shape == cache[1].shape == (batch, 4, i + 1, 4)
            assert cache[2].shape == cache[3].shape == (batch, 4, 7, 4)
            assert [cache[j].data_ptr() for j in [2, 3]] == cross_storage
            outputs.append(output)
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected, rtol=1e-5, atol=1e-6
    )


def test_legacy_raw_state_cache_is_rejected():
    layer = CachedTransformerDecoderLayer(16, 4, 32).eval()
    legacy = (torch.randn(1, 3, 16), torch.randn(1, 3, 16))
    with pytest.raises(ValueError, match="four projected cache tensors"):
        layer(torch.randn(1, 1, 16), torch.randn(1, 7, 16), past_key_value=legacy)


def uncached_generate(model, images, tokenizer, max_length):
    memory = model.encode_images(images)
    ids = torch.full((images.shape[0], 1), tokenizer.bos_token_id, dtype=torch.long)
    for _ in range(max_length):
        output = model(input_ids=ids, encoder_outputs=memory, use_cache=False)
        token = output.logits[:, -1:].argmax(dim=-1)
        ids = torch.cat([ids, token], dim=1)
        if (token == tokenizer.eos_token_id).all():
            break
    output = model(input_ids=ids, encoder_outputs=memory, use_cache=False)
    boxes = None
    positions = [
        torch.isin(seq, torch.tensor(model.data_cells)).nonzero().flatten()
        for seq in ids
    ]
    if any(len(p) for p in positions):
        boxes = torch.zeros(images.shape[0], max(map(len, positions)), 4)
        for image, pos in enumerate(positions):
            if len(pos):
                boxes[image, : len(pos)] = model.bbox_head(
                    output.hidden_states[image, pos],
                    memory["last_hidden_state"][image : image + 1],
                    torch.zeros(len(pos), dtype=torch.long),
                )
    return ids, boxes


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("cell_tokens", [False, True])
def test_generate_matches_uncached_reference(batch, cell_tokens):
    torch.manual_seed(30)
    config = TableFormerV2Config(
        embed_dim=16,
        num_heads=4,
        ff_dim=32,
        num_decoder_layers=2,
        vocab_size=4,
        data_cells=[2],
    )
    with patch(
        "docling_ibm_models.tableformer_v2.model.efficientnet_v2_s",
        return_value=nn.Identity(),
    ):
        model = TableFormerV2(config).eval()
    model.encode_images = lambda images: {"last_hidden_state": images}
    tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=1)
    images = torch.randn(batch, 9, 16)
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.zero_()
        model.output_projection.bias[2 if cell_tokens else 1] = 10
        ids, boxes = uncached_generate(model, images, tokenizer, max_length=8)
        actual = model.generate(images, tokenizer, max_length=8)
    torch.testing.assert_close(actual["generated_ids"], ids, rtol=0, atol=0)
    if boxes is None:
        assert actual["predicted_bboxes"] is None
    else:
        torch.testing.assert_close(
            actual["predicted_bboxes"], boxes, rtol=1e-5, atol=1e-6
        )
