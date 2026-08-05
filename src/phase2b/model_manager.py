from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time   


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    HuggingFace-based local model manager.

    Loads any HuggingFace causal-LM locally and exposes the same
    generate() / cleanup() interface used by CrossModelProber.

    COMPATIBLE WITH: Any HuggingFace causal-LM (Qwen, Llama, Mistral, Gemma, etc.)
    """

    def __init__(self, model_name: str, device: str = "cuda", debug: bool = False):
        """
        Load a HuggingFace causal-LM and its tokenizer.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B-Instruct")
            device: "cuda", "cpu", or "auto"  (auto = device_map="auto")
            debug: Enable debug logging
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.debug = debug

        print(f"⏳ Loading HuggingFace model: {model_name}  (device={device})")

        # ── tokenizer ──────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── model ──────────────────────────────────────────────────────────────
        load_kwargs: dict = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        # Resolve the actual device the model lives on (for input tensors)
        try:
            self._infer_device = next(self.model.parameters()).device
        except StopIteration:
            self._infer_device = torch.device("cpu")

        print(f"✅ HuggingFace model loaded: {model_name}  [{self._infer_device}]")

    # ──────────────────────────────────────────────────────────────────────────
    # Repetition detection
    # ──────────────────────────────────────────────────────────────────────────
    def detect_repetition(self, text: str, threshold: int = 3) -> bool:
        """Detect if model is stuck repeating (same as original)."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < threshold:
            return False

        # Check last N sentences
        last_n = sentences[-threshold:]
        if len(set(last_n)) == 1:
            return True

        # Check for phrase repetition
        repetitive_phrases = [
            'The question asked',
            'The answer is',
            'Therefore',
        ]
        for phrase in repetitive_phrases:
            if text.count(phrase) > 3:
                return True

        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────
    def generate(self, prompt: str, max_new_tokens: int = 300) -> Tuple[str, float, int, int]:
        """
        Generate a response using a locally-loaded HuggingFace model.

        Applies chat template when available (instruct models), otherwise uses
        the raw prompt.

        Args:
            prompt: Input prompt (plain text or already-formatted)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple[str, float, int, int]:
                (full_text, generation_time, input_tokens, output_tokens)
            where full_text = prompt + generated_text  (same as original API version)
        """
        import torch

        start_time = time.time()

        try:
            # ── tokenise ──────────────────────────────────────────────────────
            # Use chat template for instruct models when available
            if (hasattr(self.tokenizer, "chat_template")
                    and self.tokenizer.chat_template is not None):
                messages = [{"role": "user", "content": prompt}]
                encoded_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                encoded_prompt = prompt

            inputs = self.tokenizer(
                encoded_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self._infer_device)

            input_tokens: int = inputs["input_ids"].shape[-1]

            # ── generate ─────────────────────────────────────────────────────
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generation_time: float = time.time() - start_time

            # ── decode ────────────────────────────────────────────────────────
            # Slice off the input tokens so we only decode the new tokens
            new_ids = output_ids[0][input_tokens:]
            output_tokens: int = new_ids.shape[-1]

            generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            # Combine prompt + generation (identical interface to API version)
            full_text = prompt + generated_text

            if self.debug:
                print(f" Generated {output_tokens} tokens in {generation_time:.2f}s")

            # Check for repetition
            if self.detect_repetition(generated_text):
                if self.debug:
                    print(f" REPETITION DETECTED")

            return full_text, generation_time, input_tokens, output_tokens

        except Exception as e:
            print(f" Generation Error: {e}")
            # Return empty response on error (same contract as API version)
            return prompt, 0.0, 0, 0

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────
    def cleanup(self):
        """
        Free GPU memory by deleting the model and tokenizer and calling
        torch.cuda.empty_cache().
        """
        import torch
        import gc

        print(f" Cleaning up model: {self.model_name}")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f" Cleanup complete: {self.model_name}")

