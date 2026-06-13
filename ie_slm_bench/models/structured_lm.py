from __future__ import annotations

import json
import time

import pandas as pd
import torch
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

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

    def load(self) -> None:
        if self.backend_kind == "gemma":
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.tokenizer = self.processor.tokenizer
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        del self.processor
        self.model = None
        self.tokenizer = None
        self.processor = None
        torch.cuda.empty_cache()

    def _format_prompt(self, text: str, model_cls: type[BaseModel]) -> str:
        user_prompt = build_user_prompt(text, model_cls)
        if self.backend_kind == "nuextract":
            template = {
                "text": text,
                "schema": json.loads(model_cls.model_json_schema()),
            }
            return (
                "# Template:\n"
                f"{json.dumps(template, ensure_ascii=False)}\n"
                "# Output:"
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if self.backend_kind == "gemma" and self.processor is not None:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        if self.backend_kind == "gemma" and self.processor is not None:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    def predict_frame(
        self,
        frame: pd.DataFrame,
        benchmark: str,
        max_new_tokens: int,
    ) -> pd.DataFrame:
        model_cls = benchmark_schema(benchmark)
        records = []
        started = time.time()
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"{self.model_id}:{benchmark}"):
            prompt = self._format_prompt(row["text"], model_cls)
            pred_raw = self._generate_one(prompt, max_new_tokens=max_new_tokens)
            records.append(
                {
                    "benchmark": benchmark,
                    "model_id": self.model_id,
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "gold_json": row["gold_json"],
                    "pred_raw": pred_raw,
                    "latency_sec": time.time() - started,
                }
            )
        return pd.DataFrame(records)
