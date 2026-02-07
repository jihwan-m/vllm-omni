# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Chunked stage input processors for Moshi streaming pipeline (Phase 2).

These processors are used with async_chunk=true to enable incremental
transfer of audio codes from the dialogue stage to the Mimi decoder,
allowing audio output to start before full generation completes.

Moshi operates at 12.5Hz frame rate (80ms per temporal step).
Each step produces 8 audio codebook tokens via the depth decoder.
We accumulate CHUNK_SIZE frames before forwarding to Mimi for decoding.
"""
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Moshi frame rate: 12.5 Hz (one temporal step every 80ms)
# Each temporal step produces 8 RVQ codebook tokens
MOSHI_NUM_CODEBOOKS = 8

# Number of frames to accumulate before sending a chunk to Mimi decoder.
# 25 frames = 2 seconds of audio. This matches the Qwen3-Omni pattern
# and provides a good balance between latency and decode efficiency.
MOSHI_CHUNK_SIZE = 25


def dialogue_to_mimi_async_chunk(
    connector,
    pooling_output: dict,
    request,
) -> dict | None:
    """Accumulate audio code chunks from dialogue stage for Mimi decoding.

    Called by the scheduler's put_chunk() after each dialogue generation step.
    Returns None to defer (accumulate more codes), or a dict payload to send.

    The dialogue stage emits per-step pooling output containing:
      - "audio_codes": list[int] of length 8 (one codebook token per RVQ level)
      - "text_token": int (the text token for this step)

    We accumulate codes until we have CHUNK_SIZE frames, then flatten to
    [8 * chunk_size] for the Mimi decoder (same format as non-streaming).

    Args:
        connector: OmniConnectorBase with request state tracking
        pooling_output: Dict from the dialogue stage's pooler
        request: The Request object with request ID and status

    Returns:
        None to defer, or dict with "code_predictor_codes" and "finished"
    """
    request_id = request.external_req_id

    # Extract per-step audio codes from pooling output
    audio_codes = pooling_output.get("audio_codes")
    if audio_codes is None:
        return None

    # Convert to list if tensor
    if isinstance(audio_codes, torch.Tensor):
        audio_codes = audio_codes.tolist()

    # Ensure we have exactly num_codebooks codes
    if isinstance(audio_codes, (list, tuple)):
        if len(audio_codes) != MOSHI_NUM_CODEBOOKS:
            logger.warning(
                "Expected %d audio codes, got %d for request %s",
                MOSHI_NUM_CODEBOOKS, len(audio_codes), request_id,
            )
            return None
    else:
        logger.warning("Unexpected audio_codes type: %s", type(audio_codes))
        return None

    # Accumulate codes for this request
    connector.code_prompt_token_ids[request_id].append(audio_codes)
    accumulated = len(connector.code_prompt_token_ids[request_id])

    is_finished = request.is_finished()
    at_chunk_boundary = (accumulated % MOSHI_CHUNK_SIZE == 0)

    if not at_chunk_boundary and not is_finished:
        return None  # Wait for more codes

    # Determine how many frames to include in this chunk
    chunk_length = accumulated % MOSHI_CHUNK_SIZE
    if chunk_length == 0:
        chunk_length = MOSHI_CHUNK_SIZE

    # Get the codes for this chunk: last chunk_length frames
    codes = connector.code_prompt_token_ids[request_id][-chunk_length:]

    # Stack to [chunk_length, 8], transpose to [8, chunk_length], flatten
    codes_tensor = torch.tensor(codes, dtype=torch.long)  # [chunk_length, 8]
    flat_codes = codes_tensor.t().reshape(-1).tolist()  # [8 * chunk_length]

    chunk_id = connector.put_requests[request_id]

    logger.debug(
        "Moshi chunk %d for request %s: %d frames (%d codes), finished=%s",
        chunk_id, request_id, chunk_length, len(flat_codes), is_finished,
    )

    return {
        "code_predictor_codes": flat_codes,
        "finished": bool(is_finished),
        "chunk_index": chunk_id,
        "num_frames": chunk_length,
    }
