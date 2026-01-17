import sys
import os
import json
import logging
import gc
import torch
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT
from test_cases import TEST_CASES
from metrics import calculate_persona_generation_metrics
from report_generator import generate_evaluation_report
from model_interface import RealModelInterface

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Evaluator")


def cleanup_gpu_memory():
    """Forces garbage collection and clears CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    logger.info("Starting Hybrid LLM Persona Benchmark...")

    models_config = []

    # 1. API: OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "sk-" in openai_key:
        models_config.append(("GPT-4o", "gpt-4o", openai_key, None, False))

    # 2. API: Ollama/Local Server (Already running)
    local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
    target_ollama = os.getenv("TARGET_LOCAL_MODELS", "").split(",")
    for m in target_ollama:
        if m.strip():
            # (Friendly Name, Model ID, Key, URL, is_local_hf)
            models_config.append(
                (f"Ollama-{m.strip()}", m.strip(), "ollama", local_url, False)
            )

    # 3. Local: HuggingFace (Direct Download & Load)
    # Define models in .env as TARGET_HF_MODELS=google/gemma-2-9b-it,mistralai/Mistral-7B-v0.1
    target_hf = os.getenv("TARGET_HF_MODELS", "").split(",")
    for m in target_hf:
        if m.strip():
            friendly = m.split("/")[-1]  # e.g., gemma-2-9b-it
            models_config.append(
                (f"HF-{friendly}", m.strip(), None, None, True)
            )  # is_local_hf = True

    if not models_config:
        logger.error(
            "No models configured. Check .env (OPENAI_API_KEY, TARGET_LOCAL_MODELS, TARGET_HF_MODELS)."
        )
        return

    all_results = []

    for friendly_name, model_id, api_key, base_url, is_local_hf in models_config:
        logger.info(
            f"Initializing {friendly_name} (ID: {model_id}, LocalHF: {is_local_hf})..."
        )
        model = None
        try:
            model = RealModelInterface(
                model_id, api_key, base_url, is_local=is_local_hf
            )
        except Exception as e:
            logger.error(f"Failed to load {friendly_name}: {e}")
            continue

        logger.info(f"Evaluating {friendly_name}...")
        for case in TEST_CASES:
            # logger.info(f" > Case {case['id']}...")
            try:
                output = model.generate(SYSTEM_PROMPT, case["input"])

                # Metric Calculation
                metrics = calculate_persona_generation_metrics(
                    case["input"], output["text"]
                )

                # Handle JSON parsing for report
                parsed_output = {}
                try:
                    parsed_output = json.loads(output["text"])
                except:
                    pass

                all_results.append(
                    {
                        "test_id": case["id"],
                        "case_type": case["case_type"],
                        "model_name": friendly_name,
                        "input": case["input"],
                        "output": parsed_output if parsed_output else output["text"],
                        "metrics": metrics,
                        "execution_time": output["latency"],
                        "ttft": output["ttft"],
                        "input_tokens": output["input_tokens"],
                        "output_tokens": output["output_tokens"],
                    }
                )
            except Exception as e:
                logger.error(f"Error during generation for {friendly_name}: {e}")

        # Cleanup Memory immediately after model usage
        logger.info(f"Unloading {friendly_name} and cleaning up memory...")
        del model
        cleanup_gpu_memory()

    report_path = generate_evaluation_report(all_results, output_path="./reports")
    logger.info(f"Done! Report: {report_path}")


if __name__ == "__main__":
    main()
