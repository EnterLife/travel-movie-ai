"""Isolated ACE-Step worker used by the model-specific virtual environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

RESULT_PREFIX = "TRAVELMOVIEAI_RESULT="


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    arguments = parser.parse_args()
    request_path = Path(arguments.request)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    os.environ["ACESTEP_PROJECT_ROOT"] = request["project_root"]
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = request["checkpoint_dir"]

    from acestep.handler import AceStepHandler  # type: ignore[import-not-found]
    from acestep.inference import (  # type: ignore[import-not-found]
        GenerationConfig,
        GenerationParams,
        generate_music,
    )

    handler = AceStepHandler()
    status, initialized = handler.initialize_service(
        project_root=request["project_root"],
        config_path=request["config_path"],
        device=request["device"],
        offload_to_cpu=request["offload_to_cpu"],
        offload_dit_to_cpu=request["offload_dit_to_cpu"],
    )
    if not initialized:
        raise RuntimeError(f"ACE-Step DiT initialization failed: {status}")

    lora_path = request.get("lora_path")
    if lora_path:
        lora_status = handler.load_lora(lora_path)
        if not bool(getattr(handler, "lora_loaded", False)):
            raise RuntimeError(f"ACE-Step LoRA initialization failed: {lora_status}")
        handler.set_use_lora(True)
        handler.set_lora_scale(float(request.get("lora_strength", 0.7)))

    llm_handler: Any = None
    if request.get("thinking"):
        from acestep.llm_inference import LLMHandler  # type: ignore[import-not-found]

        llm_handler = LLMHandler()
        lm_status, lm_initialized = llm_handler.initialize(
            checkpoint_dir=request["checkpoint_dir"],
            lm_model_path=request["lm_model"],
            backend=request["lm_backend"],
            device=request["device"],
            offload_to_cpu=True,
        )
        if not lm_initialized:
            raise RuntimeError(f"ACE-Step LM initialization failed: {lm_status}")

    generated: list[dict[str, object]] = []
    seeds = [int(value) for value in request["seeds"]]
    for index, seed in enumerate(seeds):
        print(f"TravelMovieAI candidate {index + 1}/{len(seeds)}", flush=True)
        params = GenerationParams(
            task_type="text2music",
            reference_audio=request.get("reference_audio"),
            caption=request["prompt"],
            lyrics="[Instrumental]",
            instrumental=True,
            bpm=int(request["bpm"]),
            keyscale=request["keyscale"],
            timesignature=request["timesignature"],
            duration=float(request["duration_seconds"]),
            inference_steps=int(request["inference_steps"]),
            seed=seed,
            guidance_scale=float(request["guidance_scale"]),
            use_adg=bool(request["use_adg"]),
            shift=float(request["shift"]),
            audio_cover_strength=float(request.get("reference_strength", 0.2)),
            thinking=bool(request.get("thinking")),
            use_cot_metas=bool(request.get("thinking")),
            use_cot_caption=bool(request.get("thinking")),
            use_cot_lyrics=False,
            use_cot_language=False,
        )
        config = GenerationConfig(
            batch_size=1,
            seeds=[seed],
            use_random_seed=False,
            audio_format="wav32",
        )
        result = generate_music(
            handler,
            llm_handler,
            params,
            config,
            save_dir=request["output_dir"],
        )
        if not result.success or not result.audios:
            raise RuntimeError(result.error or "ACE-Step did not return an audio candidate")
        audio = result.audios[0]
        generated.append({"path": str(Path(audio["path"]).resolve()), "seed": seed})

    print(RESULT_PREFIX + json.dumps({"candidates": generated}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
