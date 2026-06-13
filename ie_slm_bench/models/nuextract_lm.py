from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from ie_slm_bench.env import configure_torch_runtime

configure_torch_runtime()

import torch
from pydantic import ValidationError
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from ie_slm_bench.config import (
    BENCHMARK,
    INFER_RETRY_MAX_NEW_TOKENS,
    LOAD_IN_4BIT,
    MAX_INPUT_CHARS,
    SAVE_EVERY_N,
)
from ie_slm_bench.nuextract_template import BANK_CLIENT_NUEXTRACT_TEMPLATE
from ie_slm_bench.parsers import extract_json_object
from ie_slm_bench.prompts import truncate_text
from schemas.bank_client import BankClientExtraction


class NuExtractBackend:
    def __init__(self, model_id: str, batch_size: int = 8):
        self.model_id = model_id
        self.batch_size = batch_size
        self.hf_model = None
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
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True,
        )
        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

    def unload(self) -> None:
        del self.hf_model
        del self.processor
        self.hf_model = None
        self.processor = None
        torch.cuda.empty_cache()

    def _format_batch_texts(self, texts: list[str]) -> list[str]:
        messages_batch = [
            [{"role": "user", "content": truncate_text(text, MAX_INPUT_CHARS)}]
            for text in texts
        ]
        return [
            self.processor.tokenizer.apply_chat_template(
                messages_batch[index],
                template=BANK_CLIENT_NUEXTRACT_TEMPLATE,
                tokenize=False,
                add_generation_prompt=True,
            )
            for index in range(len(messages_batch))
        ]

    def _parse_output(self, raw_text: str) -> BankClientExtraction:
        payload = extract_json_object(raw_text)
        return BankClientExtraction.model_validate(payload)

    def _generate_batch(self, texts: list[str], max_new_tokens: int) -> list[str]:
        prompt_texts = self._format_batch_texts(texts)
        inputs = self.processor(
            text=prompt_texts,
            images=None,
            padding=True,
            return_tensors="pt",
        ).to(self.hf_model.device)

        generated_ids = self.hf_model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
        trimmed_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        predictions: list[str] = []
        for index, raw_text in enumerate(output_texts):
            try:
                parsed = self._parse_output(raw_text)
            except (ValidationError, json.JSONDecodeError):
                single_inputs = self.processor(
                    text=[prompt_texts[index]],
                    images=None,
                    padding=True,
                    return_tensors="pt",
                ).to(self.hf_model.device)
                retry_ids = self.hf_model.generate(
                    **single_inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=INFER_RETRY_MAX_NEW_TOKENS,
                )
                retry_trimmed = retry_ids[0, single_inputs.input_ids.shape[1] :]
                retry_text = self.processor.batch_decode(
                    [retry_trimmed],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                try:
                    parsed = self._parse_output(retry_text)
                except (ValidationError, json.JSONDecodeError):
                    print("Warning: NuExtract extraction failed, using empty prediction")
                    parsed = BankClientExtraction()
            predictions.append(parsed.model_dump_json(by_alias=True, exclude_none=True))
        return predictions

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
            print(
                f"{self.model_id}:{BENCHMARK}, batch_size={self.batch_size}, "
                "backend=nuextract"
            )

        batch_starts = range(0, len(pending), self.batch_size)
        for batch_index, batch_start in enumerate(
            tqdm(batch_starts, desc=f"{self.model_id}:{BENCHMARK}")
        ):
            batch_rows = pending.iloc[batch_start : batch_start + self.batch_size]
            started = time.time()
            texts = [row["text"] for _, row in batch_rows.iterrows()]
            pred_raw_list = self._generate_batch(texts, max_new_tokens=max_new_tokens)
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
