# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mock vllm dependencies for Moshi model unit tests.

This conftest.py registers mock modules in sys.modules BEFORE pytest
collects test files, allowing Moshi model code to be imported without
the full vllm package installed.
"""
import logging
import os
import sys
from types import ModuleType
from typing import Any, TypedDict
from unittest.mock import MagicMock

import torch.nn as nn

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_VLLM_OMNI_DIR = os.path.join(_ROOT, "vllm_omni")


def _register_pkg(name: str, path: str | None = None, attrs: dict | None = None):
    """Register a package module in sys.modules without executing __init__.py."""
    if name in sys.modules:
        return sys.modules[name]
    mod = ModuleType(name)
    mod.__package__ = name
    mod.__path__ = [path] if path else []
    if path:
        mod.__file__ = os.path.join(path, "__init__.py")
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _register_mod(name: str, attrs: dict | None = None):
    """Register a non-package module in sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    mod = ModuleType(name)
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Only set up mocks if vllm is NOT installed
_vllm_available = True
try:
    import vllm  # noqa: F401
except ImportError:
    _vllm_available = False

if not _vllm_available:
    # ===========================================================
    # Stub classes for vllm types used in inheritance / as bases
    # ===========================================================

    class _UnquantizedLinear(nn.Module):
        """Stub for vllm.model_executor.layers.linear.UnquantizedLinear."""

        pass

    class _ColumnParallelLinear(nn.Module):
        """Stub for vllm.model_executor.layers.linear.ColumnParallelLinear.

        Wraps nn.Linear but returns (output, bias) tuples like vLLM's
        parallel layers. Accepts and ignores quant_config/prefix kwargs.
        """

        def __init__(self, input_size, output_size, bias=True, **kwargs):
            super().__init__()
            self._linear = nn.Linear(input_size, output_size, bias=bias)
            # Expose .weight (and .bias) at the top level so
            # named_parameters() matches HF checkpoint paths.
            self.weight = self._linear.weight
            if bias:
                self.bias = self._linear.bias

        def forward(self, x):
            return self._linear(x), None

    class _RowParallelLinear(nn.Module):
        """Stub for vllm.model_executor.layers.linear.RowParallelLinear.

        Same contract as _ColumnParallelLinear.
        """

        def __init__(self, input_size, output_size, bias=True, **kwargs):
            super().__init__()
            self._linear = nn.Linear(input_size, output_size, bias=bias)
            self.weight = self._linear.weight
            if bias:
                self.bias = self._linear.bias

        def forward(self, x):
            return self._linear(x), None

    class _SupportsPP:
        """Stub for vllm.model_executor.models.interfaces.SupportsPP."""

        pass

    class _IntermediateTensors:
        """Stub for vllm.sequence.IntermediateTensors."""

        pass

    # TypedDicts used as base classes in vllm_omni.inputs.data
    class _TextPrompt(TypedDict, total=False):
        prompt: str
        multi_modal_data: Any
        mm_processor_kwargs: Any

    class _TokensPrompt(TypedDict, total=False):
        prompt_token_ids: list[int]
        multi_modal_data: Any
        mm_processor_kwargs: Any

    class _EmbedsPrompt(TypedDict, total=False):
        prompt_embeds: Any

    class _TokenInputs(TypedDict, total=False):
        type: str
        prompt_token_ids: list[int]

    # ===========================================================
    # Mock vllm.* modules
    # ===========================================================

    # vllm (top-level package)
    _register_pkg(
        "vllm",
        attrs={
            "PromptType": type("PromptType", (), {}),
            "SamplingParams": type("SamplingParams", (), {}),
        },
    )

    # vllm.config hierarchy
    _register_pkg(
        "vllm.config",
        attrs={
            "VllmConfig": type("VllmConfig", (), {}),
            "ModelConfig": type("ModelConfig", (), {}),
            "config": MagicMock(),
        },
    )
    _register_mod("vllm.config.lora", {"LoRAConfig": type("LoRAConfig", (), {})})
    _register_mod(
        "vllm.config.model",
        {
            "_RUNNER_CONVERTS": {},
            "_get_and_verify_dtype": MagicMock(),
            "get_served_model_name": MagicMock(),
            "ModelConfig": type("ModelConfig", (), {}),
            "config": MagicMock(),
        },
    )
    _register_mod(
        "vllm.config.multimodal",
        {
            "MMCacheType": MagicMock(),
            "MMEncoderTPMode": MagicMock(),
            "MultiModalConfig": type("MultiModalConfig", (), {}),
        },
    )
    _register_mod("vllm.config.pooler", {"PoolerConfig": type("PoolerConfig", (), {})})

    # vllm.logger
    _register_mod(
        "vllm.logger", {"init_logger": lambda name: logging.getLogger(name)}
    )

    # vllm.model_executor hierarchy
    _register_pkg("vllm.model_executor")
    _register_pkg("vllm.model_executor.layers")
    _register_mod(
        "vllm.model_executor.layers.linear",
        {
            "UnquantizedLinear": _UnquantizedLinear,
            "ColumnParallelLinear": _ColumnParallelLinear,
            "RowParallelLinear": _RowParallelLinear,
        },
    )
    _register_pkg("vllm.model_executor.models")
    _register_mod(
        "vllm.model_executor.models.interfaces", {"SupportsPP": _SupportsPP}
    )
    _register_mod(
        "vllm.model_executor.models.utils",
        {"AutoWeightsLoader": MagicMock(), "WeightsMapper": MagicMock()},
    )
    _register_mod(
        "vllm.model_executor.models.registry",
        {
            "_VLLM_MODELS": {},
            "_LazyRegisteredModel": type(
                "_LazyRegisteredModel", (), {"__init__": lambda self, **kw: None}
            ),
            "_ModelRegistry": type(
                "_ModelRegistry", (), {"__init__": lambda self, d: None}
            ),
        },
    )

    # vllm.sequence
    _register_mod("vllm.sequence", {"IntermediateTensors": _IntermediateTensors})

    # vllm.v1 hierarchy
    _register_pkg("vllm.v1")
    _register_pkg("vllm.v1.sample")
    _register_mod(
        "vllm.v1.sample.metadata",
        {"SamplingMetadata": type("SamplingMetadata", (), {})},
    )
    _register_pkg("vllm.v1.attention")
    _register_pkg("vllm.v1.attention.backends")
    _register_mod(
        "vllm.v1.attention.backends.registry", {"AttentionBackendEnum": MagicMock()}
    )

    # vllm.inputs hierarchy
    _register_pkg("vllm.inputs", attrs={"TextPrompt": _TextPrompt})
    _register_mod(
        "vllm.inputs.data",
        {
            "EmbedsPrompt": _EmbedsPrompt,
            "TextPrompt": _TextPrompt,
            "TokenInputs": _TokenInputs,
            "TokensPrompt": _TokensPrompt,
        },
    )

    # vllm.lora hierarchy
    _register_pkg("vllm.lora")
    _register_mod(
        "vllm.lora.request", {"LoRARequest": type("LoRARequest", (), {})}
    )

    # vllm.platforms
    _register_mod("vllm.platforms", {"current_platform": MagicMock()})

    # vllm.utils
    _register_mod(
        "vllm.utils",
        {"random_uuid": lambda: "mock-uuid-1234"},
    )

    # vllm.transformers_utils hierarchy
    _register_pkg("vllm.transformers_utils")
    _register_mod(
        "vllm.transformers_utils.config",
        {
            "get_config": MagicMock(),
            "get_hf_image_processor_config": MagicMock(),
            "get_hf_text_config": MagicMock(),
            "get_pooling_config": MagicMock(),
        },
    )
    _register_mod(
        "vllm.transformers_utils.gguf_utils",
        {"is_gguf": MagicMock(), "maybe_patch_hf_config_from_gguf": MagicMock()},
    )
    _register_mod(
        "vllm.transformers_utils.utils", {"maybe_model_redirect": MagicMock()}
    )

    # vllm.outputs and vllm.v1.outputs
    _register_mod(
        "vllm.outputs",
        {"RequestOutput": type("RequestOutput", (), {})},
    )
    _register_mod(
        "vllm.v1.outputs",
        {"ModelRunnerOutput": type("ModelRunnerOutput", (), {})},
    )
    _register_pkg("vllm.v1.core")
    _register_pkg("vllm.v1.core.sched")
    _register_mod(
        "vllm.v1.core.sched.output",
        {"SchedulerOutput": type("SchedulerOutput", (), {})},
    )
    _register_pkg("vllm.v1.request")
    _register_mod(
        "vllm.v1.request",
        {
            "Request": type("Request", (), {}),
            "RequestStatus": type("RequestStatus", (), {"FINISHED_STOPPED": 0}),
        },
    )

    # ===========================================================
    # Pre-register vllm_omni packages to bypass __init__.py chains
    # that have deep vllm dependencies (config, entrypoints, etc.)
    # ===========================================================
    _register_pkg("vllm_omni", _VLLM_OMNI_DIR)
    _register_pkg(
        "vllm_omni.model_executor",
        os.path.join(_VLLM_OMNI_DIR, "model_executor"),
    )
    _register_pkg(
        "vllm_omni.model_executor.models",
        os.path.join(_VLLM_OMNI_DIR, "model_executor", "models"),
    )

    # Pre-register distributed hierarchy to bypass __init__.py
    # chains that import shm_connector → stage_utils → omegaconf.
    _register_pkg(
        "vllm_omni.distributed",
        os.path.join(_VLLM_OMNI_DIR, "distributed"),
    )
    _register_pkg(
        "vllm_omni.distributed.omni_connectors",
        os.path.join(_VLLM_OMNI_DIR, "distributed", "omni_connectors"),
    )

    # Pre-register entrypoints hierarchy to bypass __init__.py
    # chains that import AsyncOmni, api_server, etc.
    _register_pkg(
        "vllm_omni.entrypoints",
        os.path.join(_VLLM_OMNI_DIR, "entrypoints"),
    )
    _register_pkg(
        "vllm_omni.entrypoints.openai",
        os.path.join(_VLLM_OMNI_DIR, "entrypoints", "openai"),
    )
    _register_pkg(
        "vllm_omni.entrypoints.openai.protocol",
        os.path.join(_VLLM_OMNI_DIR, "entrypoints", "openai", "protocol"),
    )
