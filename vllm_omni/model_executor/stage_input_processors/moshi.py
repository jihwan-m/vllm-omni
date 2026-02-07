# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for Moshi: Dialogue → Mimi Decode transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _validate_stage_inputs(stage_list, engine_input_source):
    """Validate that source stage has outputs ready."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    # engine_outputs can be a single RequestOutput or a list
    outputs = stage.engine_outputs
    if not isinstance(outputs, list):
        outputs = [outputs]

    return outputs


def dialogue_to_mimi_decode(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Transform Moshi dialogue outputs to Mimi decoder inputs.

    Dialogue stage outputs:
      - multimodal_output["audio_codes"]: torch.Tensor [T, 8] (T time steps, 8 codebooks)
      - multimodal_output["text_tokens"]: torch.Tensor [T] (text tokens for inner monologue)

    Mimi decode stage expects:
      - prompt_token_ids: flattened audio codes [8*T] (8 codebooks interleaved per frame)
    """
    dialogue_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    mimi_inputs: list[OmniTokensPrompt] = []

    for dialogue_output in dialogue_outputs:
        output = dialogue_output.outputs[0]

        # Extract audio codes from dialogue stage multimodal output
        audio_codes = output.multimodal_output.get("audio_codes")
        if audio_codes is None:
            logger.error("Dialogue stage output missing 'audio_codes' in multimodal_output")
            # Return empty prompt as fallback
            mimi_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[0] * 8,
                    multi_modal_data=None,
                )
            )
            continue

        # audio_codes: [T, 8] → transpose to [8, T] → flatten to [8*T]
        if isinstance(audio_codes, torch.Tensor):
            # Transpose: [T, 8] → [8, T], then flatten row-major
            flat_codes = (
                audio_codes
                .to(torch.long)
                .T  # [8, T]
                .contiguous()
                .reshape(-1)  # [8*T]
                .cpu()
                .tolist()
            )
        else:
            # Already a list
            flat_codes = list(audio_codes)

        logger.debug(
            "dialogue_to_mimi_decode: audio_codes shape=%s, flat_codes len=%d",
            getattr(audio_codes, "shape", "unknown"),
            len(flat_codes),
        )

        mimi_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=flat_codes,
                multi_modal_data=None,
            )
        )

    return mimi_inputs
