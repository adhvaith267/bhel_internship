import json
import time
from typing import Generator, Optional
import httpx

from bhel_internship.core.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GGUF_MODEL_PATH
)
from bhel_internship.core.logger import logger


def _default_stop_sequences() -> list[str]:
    """Default stop sequences that work across common chat templates."""
    return ["\nHuman:", "\n\nHuman:", "\nUser:", "\n\nUser:", "[END]"]


class LLMEngine:
    def __init__(self):
        self.provider = LLM_PROVIDER.lower()
        self.active_model_name = "unknown"
        self._llamacpp_instance = None
        self._check_and_init_provider()

    def _is_ollama_available(self) -> bool:
        """Checks if local Ollama server is running and accessible."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name") for m in res.json().get("models", [])]
                    logger.info(f"Ollama detected with models: {models}")
                    return True
        except Exception:
            return False
        return False

    def _check_and_init_provider(self):
        if self.provider == "auto":
            if self._is_ollama_available():
                self.provider = "ollama"
                self.active_model_name = f"Ollama ({OLLAMA_MODEL})"
                logger.info(f"[bold green]✓ Using Ollama provider with model: {OLLAMA_MODEL}[/bold green]")
            else:
                self.provider = "llamacpp"
                self.active_model_name = f"llama.cpp ({GGUF_MODEL_PATH.name})"
                logger.info(f"[bold yellow]Ollama not available. Falling back to llama.cpp: {GGUF_MODEL_PATH.name}[/bold yellow]")
        elif self.provider == "ollama":
            self.active_model_name = f"Ollama ({OLLAMA_MODEL})"
            logger.info(f"[bold green]Using Ollama provider: {OLLAMA_MODEL}[/bold green]")
        else:
            self.provider = "llamacpp"
            self.active_model_name = f"llama.cpp ({GGUF_MODEL_PATH.name})"
            logger.info(f"[bold green]Using llama.cpp provider: {GGUF_MODEL_PATH.name}[/bold green]")

    def _get_llamacpp(self):
        if self._llamacpp_instance is None:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model into memory: {GGUF_MODEL_PATH}")
            self._llamacpp_instance = Llama(
                model_path=str(GGUF_MODEL_PATH),
                n_gpu_layers=20,
                n_threads=6,
                n_ctx=4096,
                verbose=False
            )
        return self._llamacpp_instance

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.2
    ) -> str:
        """Synchronously generates a completed text answer."""
        t0 = time.time()
        
        if self.provider == "ollama":
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt or "",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("response", "").strip()
                    latency = time.time() - t0
                    eval_count = data.get("eval_count", 0)
                    tok_sec = (eval_count / latency) if latency > 0 and eval_count else 0
                    logger.info(
                        f"[RAG.LLM] Generation finished in {latency:.2f}s "
                        f"({eval_count} tokens, {tok_sec:.1f} t/s)"
                    )
                    return answer
            except Exception as e:
                logger.error(f"Ollama generation error: {e}")
                raise
        else:
            return self._generate_llamacpp(prompt, max_tokens, temperature)

    def _generate_llamacpp(self, prompt: str, max_tokens: int, temperature: float) -> str:
        t0 = time.time()
        try:
            llm = self._get_llamacpp()
            res = llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=_default_stop_sequences()
            )
            text = res["choices"][0]["text"].strip()
            logger.info(f"[RAG.LLM] llama.cpp generation finished in {time.time()-t0:.2f}s")
            return text
        except Exception as e:
            logger.error(f"llama.cpp generation error: {e}")
            return f"Error generating answer: {str(e)}"

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.2
    ) -> Generator[str, None, None]:
        """Streams generation chunks token-by-token."""
        if self.provider == "ollama":
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt or "",
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            try:
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as response:
                        for line in response.iter_lines():
                            if not line:
                                continue
                            data = json.loads(line)
                            yield data.get("response", "")
                            if data.get("done", False):
                                break
            except Exception as e:
                logger.error(f"Ollama streaming error: {e}")
                yield f"[Streaming error: {e}]"
        else:
            llm = self._get_llamacpp()
            for token_data in llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stop=_default_stop_sequences()
            ):
                text_chunk = token_data["choices"][0]["text"]
                yield text_chunk
