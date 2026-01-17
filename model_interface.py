import time
import json
import re
import torch
from typing import Dict, Any, Union
from openai import OpenAI

# Try importing transformers (graceful failure if not installed)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class RealModelInterface:
    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        base_url: str = None,
        is_local: bool = False,
    ):
        """
        is_local=True: Uses HuggingFace Transformers locally.
        is_local=False: Uses OpenAI compatible API (GPT-4, Ollama, vLLM).
        """
        self.model_name = model_name
        self.is_local = is_local

        if self.is_local:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers library not found. Install it via requirements.txt"
                )

            print(
                f"Loading Local JSON Model: {model_name} (This may take a while to download if not cached)..."
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Use bfloat16 for newer models (like Phi-3.5) if supported, else float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                self.pipe = pipeline(
                    "text-generation", model=self.model, tokenizer=self.tokenizer
                )
            except Exception as e:
                print(f"Error loading local model {model_name}: {e}")
                raise e
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self, system_prompt: str, user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Routes generation to either Local HF or API.
        """
        if self.is_local:
            result = self._generate_local(system_prompt, user_input)
        else:
            result = self._generate_api(system_prompt, user_input)

        return result

    def _generate_local(
        self, system_prompt: str, user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        user_content = json.dumps(user_input, ensure_ascii=False)

        # Prepare Prompt properly using chat template
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except:
            # Fallback for models without chat template in tokenizer
            prompt_text = f"System: {system_prompt}\nUser: {user_content}\nAssistant:"

        start_time = time.time()

        try:
            outputs = self.pipe(
                prompt_text,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                return_full_text=False,
            )
            response_text = outputs[0]["generated_text"]
        except Exception as e:
            return self._error_response(str(e))

        latency = time.time() - start_time
        cleaned_text = self._clean_json_markdown(response_text)

        # Calculate tokens locally
        in_tokens = len(self.tokenizer.encode(prompt_text))
        out_tokens = len(self.tokenizer.encode(response_text))

        return {
            "text": cleaned_text,
            "ttft": 0.0,
            "latency": latency,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }

    def _generate_api(
        self, system_prompt: str, user_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        user_content = json.dumps(user_input, ensure_ascii=False)
        start_time = time.time()
        ttft = 0.0

        try:
            # Note: Streaming makes getting exact usage harder with OpenAI.
            # We will approximate or use the 'usage' field if provided in final chunk (OpenAI recently added this).
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                stream=True,
                stream_options={"include_usage": True},  # Request usage stats in stream
            )

            collected_chunks = []
            first_token_received = False
            usage_data = None

            for chunk in stream:
                if (
                    not first_token_received
                    and chunk.choices
                    and chunk.choices[0].delta.content
                ):
                    ttft = time.time() - start_time
                    first_token_received = True

                if chunk.choices and chunk.choices[0].delta.content:
                    collected_chunks.append(chunk.choices[0].delta.content)

                # Check for usage in the last chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = chunk.usage

            response_text = "".join(collected_chunks)
            if ttft == 0:
                ttft = time.time() - start_time

            # Use real usage if available, else approximate
            if usage_data:
                in_tokens = usage_data.prompt_tokens
                out_tokens = usage_data.completion_tokens
            else:
                # Approximation (fallback)
                in_tokens = len(system_prompt + user_content) // 4
                out_tokens = len(response_text) // 4

            return {
                "text": self._clean_json_markdown(response_text),
                "ttft": ttft,
                "latency": time.time() - start_time,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
            }

        except Exception as e:
            return self._error_response(str(e))

    def _clean_json_markdown(self, text: str) -> str:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.replace("```", "").strip()

    def _error_response(self, error_msg: str) -> Dict:
        return {
            "text": json.dumps({"error": error_msg}),
            "ttft": 0.0,
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
