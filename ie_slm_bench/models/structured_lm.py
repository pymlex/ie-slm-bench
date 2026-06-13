from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ie_slm_bench.config import LOAD_IN_4BIT, SAVE_EVERY_N
from ie_slm_bench.prompts import SYSTEM_PROMPT, benchmark_schema, build_user_prompt


class StructuredLmBackend:
    def __init__(
        self,
        model_id: str,
        batch_size: int = 4,
        backend_kind: str = "causal",
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.backend_kind = backend_kind
        self.model = None
        self.tokenizer = None
        self.processor = None

    def _quantization_config(self) -> BitsAndBytesConfig | None:
        if not LOAD_IN_4BIT:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    def load(self) -> None:
        quantization_config = self._quantization_config()
        model_kwargs = {
            "device_map": "auto",
            "attn_implementation": "sdpa",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = torch.bfloat16

        if self.backend_kind == "gemma":
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            self.tokenizer = self.processor.tokenizer
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **model_kwargs,
        )

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        del self.processor
        self.model = None
        self.tokenizer = None
        self.processor = None
        torch.cuda.empty_cache()

    def _format_prompt(self, text: str, benchmark: str) -> str:
        if self.backend_kind == "nuextract":
            model_cls = benchmark_schema(benchmark)
            template = {
                "text": text,
                "schema": json.loads(model_cls.model_json_schema()),
            }
            return (
                "# Template:\n"
                f"{json.dumps(template, ensure_ascii=False)}\n"
                "# Output:"
            )

        user_prompt = build_user_prompt(text, benchmark)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.backend_kind == "causal" and "qwen3" in self.model_id.lower():
            template_kwargs["enable_thinking"] = False

        if self.backend_kind == "gemma" and self.processor is not None:
            return self.processor.apply_chat_template(messages, **template_kwargs)
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        if self.backend_kind == "gemma" and self.processor is not None:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    def predict_frame(
        self,
        frame: pd.DataFrame,
        benchmark: str,
        max_new_tokens: int,
        pred_path: Path | None = None,
    ) -> pd.DataFrame:
        done_ids: set[int] = set()
        records: list[dict] = []
        if pred_path is not None and pred_path.exists():
            existing = pd.read_csv(pred_path)
            records = existing.to_dict(orient="records")
            done_ids = {int(row["doc_id"]) for row in records}

        pending = frame[~frame["doc_id"].isin(done_ids)].reset_index(drop=True)
        if done_ids:
            print(f"Resuming {self.model_id}:{benchmark}, skipped {len(done_ids)} completed docs")

        for offset in tqdm(range(len(pending)), desc=f"{self.model_id}:{benchmark}"):
            row = pending.iloc[offset]
            started = time.time()
            prompt = self._format_prompt(row["text"], benchmark)
            pred_raw = self._generate_one(prompt, max_new_tokens=max_new_tokens)
            records.append(
                {
                    "benchmark": benchmark,
                    "model_id": self.model_id,
                    "doc_id": int(row["doc_id"]),
                    "text": row["text"],
                    "gold_json": row["gold_json"],
                    "pred_raw": pred_raw,
                    "latency_sec": time.time() - started,
                }
            )
            if pred_path is not None and (offset + 1) % SAVE_EVERY_N == 0:
                pd.DataFrame(records).to_csv(pred_path, index=False)

        result = pd.DataFrame(records)
        if pred_path is not None:
            result.to_csv(pred_path, index=False)
        return result
