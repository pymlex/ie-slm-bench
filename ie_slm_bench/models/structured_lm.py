from __future__ import annotations

import json
import time
from pathlib import Path

import outlines
import pandas as pd
from ie_slm_bench.env import configure_torch_runtime

configure_torch_runtime()

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ie_slm_bench.config import BENCHMARK, LOAD_IN_4BIT, MAX_INPUT_CHARS, MAX_INPUT_TOKENS, SAVE_EVERY_N
from ie_slm_bench.parsers import parse_outlines_output
from ie_slm_bench.prompts import SYSTEM_PROMPT, build_user_prompt, output_schema
from schemas.bank_client import BankClientExtraction


class StructuredLmBackend:
    def __init__(
        self,
        model_id: str,
        batch_size: int = 8,
        backend_kind: str = "causal",
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.backend_kind = backend_kind
        self.hf_model = None
        self.tokenizer = None
        self.outlines_model = None
        self.extraction_generator = None

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
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.hf_model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.outlines_model = outlines.from_transformers(self.hf_model, self.tokenizer)
        self.extraction_generator = outlines.Generator(self.outlines_model, BankClientExtraction)

    def unload(self) -> None:
        del self.hf_model
        del self.tokenizer
        del self.outlines_model
        del self.extraction_generator
        self.hf_model = None
        self.tokenizer = None
        self.outlines_model = None
        self.extraction_generator = None
        torch.cuda.empty_cache()

    def _format_prompt(self, text: str) -> str:
        user_prompt = build_user_prompt(text, MAX_INPUT_CHARS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.backend_kind in {"causal", "lfm", "nuextract"} and "qwen3" in self.model_id.lower():
            template_kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        raw_list = self.extraction_generator.batch(prompts, max_new_tokens=max_new_tokens)
        if not isinstance(raw_list, list):
            raw_list = [raw_list]
        parsed_list = [parse_outlines_output(raw, BankClientExtraction) for raw in raw_list]
        return [
            parsed.model_dump_json(by_alias=True, exclude_none=True)
            for parsed in parsed_list
        ]

    def predict_frame(
        self,
        frame: pd.DataFrame,
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
            print(
                f"Resuming {self.model_id}:{BENCHMARK}, "
                f"skipped {len(done_ids)} completed docs, batch_size={self.batch_size}"
            )
        else:
            print(f"{self.model_id}:{BENCHMARK}, batch_size={self.batch_size}, backend=outlines")

        batch_starts = range(0, len(pending), self.batch_size)
        for batch_index, batch_start in enumerate(
            tqdm(batch_starts, desc=f"{self.model_id}:{BENCHMARK}")
        ):
            batch_rows = pending.iloc[batch_start : batch_start + self.batch_size]
            started = time.time()
            prompts = [self._format_prompt(row["text"]) for _, row in batch_rows.iterrows()]
            pred_raw_list = self._generate_batch(prompts, max_new_tokens=max_new_tokens)
            latency = time.time() - started
            per_item_latency = latency / len(batch_rows)
            for row_index, (_, row) in enumerate(batch_rows.iterrows()):
                records.append(
                    {
                        "benchmark": BENCHMARK,
                        "model_id": self.model_id,
                        "doc_id": int(row["doc_id"]),
                        "text": row["text"],
                        "gold_json": row["gold_json"],
                        "pred_raw": pred_raw_list[row_index],
                        "latency_sec": per_item_latency,
                    }
                )
            if pred_path is not None and (batch_index + 1) % SAVE_EVERY_N == 0:
                pd.DataFrame(records).to_csv(pred_path, index=False)

        result = pd.DataFrame(records)
        if pred_path is not None:
            result.to_csv(pred_path, index=False)
        return result
