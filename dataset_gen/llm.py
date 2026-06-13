from __future__ import annotations

import json

import outlines
from ie_slm_bench.env import configure_torch_runtime

configure_torch_runtime()

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ie_slm_bench.config import (
    GEN_COVERAGE_MAX_NEW_TOKENS,
    GEN_GOLD_MAX_NEW_TOKENS,
    GEN_TEXT_MAX_NEW_TOKENS,
    GENERATOR_MODEL,
)
from ie_slm_bench.parsers import parse_outlines_output
from schemas.bank_client import BankClientExtraction, GoldProfileFill

CoverageCheck = CoverageValidation


class GeneratorBackend:
    def __init__(self, model_id: str = GENERATOR_MODEL, batch_size: int = 8):
        self.model_id = model_id
        self.batch_size = batch_size
        self.tokenizer = None
        self.hf_model = None
        self.outlines_model = None
        self.gold_generator = None
        self.coverage_generator = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.outlines_model = outlines.from_transformers(self.hf_model, self.tokenizer)
        self.gold_generator = outlines.Generator(self.outlines_model, GoldProfileFill)
        self.coverage_generator = outlines.Generator(self.outlines_model, CoverageCheck)
        print(
            f"Generator {self.model_id}, batch_size={self.batch_size}, "
            f"gold_max_new_tokens={GEN_GOLD_MAX_NEW_TOKENS}, "
            f"text_max_new_tokens={GEN_TEXT_MAX_NEW_TOKENS}, backend=outlines"
        )

    def unload(self) -> None:
        del self.hf_model
        del self.tokenizer
        del self.outlines_model
        del self.gold_generator
        del self.coverage_generator
        self.hf_model = None
        self.tokenizer = None
        self.outlines_model = None
        self.gold_generator = None
        self.coverage_generator = None
        torch.cuda.empty_cache()

    def _chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _parse_batch(self, raw_list: list | str, model_cls: type[BaseModel]) -> list[BaseModel]:
        if not isinstance(raw_list, list):
            raw_list = [raw_list]
        return [parse_outlines_output(raw, model_cls) for raw in raw_list]

    def _gold_prompt(self, spec: dict) -> str:
        avoid = ", ".join(spec["used_surnames"]) if spec["used_surnames"] else "нет"
        prefill_view = {
            key: (
                value.model_dump(by_alias=True)
                if isinstance(value, BaseModel)
                else value
            )
            for key, value in spec["prefill"].items()
        }
        gender = spec["gender_hint"]
        return self._chat(
            (
                "Ты создаёшь один независимый профиль клиента российского банка. "
                "Верни полный JSON по схеме GoldProfileFill. "
                "Заполни все поля схемы уникальными реалистичными значениями. "
                "Предзаполненные поля копируй без изменений. "
                "Этот образец не связан с другими запросами в батче."
            ),
            (
                f"Уникальный ключ: {spec['diversity_key']}.\n"
                f"Номер образца: {spec['sample_id'] + 1} из {spec['total']}, слот батча: {spec['batch_slot']}.\n"
                f"Предзаполненные поля: {json.dumps(prefill_view, ensure_ascii=False)}\n"
                f"Пол для ФИО: {gender}. Регион адреса: {spec['region_hint']}. Работа: {spec['job_hint']}.\n"
                f"Не используй фамилии: {avoid}.\n"
                "ФИО, адреса с городом, улицей и домом, работа и паспортные текстовые поля уникальны."
            ),
        )

    def _client_text_prompt(self, gold: BankClientExtraction) -> str:
        gold_view = gold.model_dump(by_alias=True, exclude_none=True)
        return self._chat(
            (
                "Ты пишешь фрагмент переписки клиента с сотрудником банка на русском языке. "
                "Текст должен содержать только те данные, которые указаны в JSON. "
                "Поля, отсутствующие в JSON, не упоминай и не выдумывай. "
                "Допустимо написать, что часть данных клиент укажет позже. "
                "Пиши 3–10 предложений, без markdown."
            ),
            f"JSON клиента:\n{json.dumps(gold_view, ensure_ascii=False)}",
        )

    def _coverage_prompt(self, text: str, gold: BankClientExtraction) -> str:
        gold_view = gold.model_dump(by_alias=True, exclude_none=True)
        return self._chat(
            (
                "Проверь, что каждое непустое поле из JSON встречается в тексте клиента. "
                "Верни JSON: all_present, missing_fields, justification. "
                "justification — короткое обоснование на русском языке."
            ),
            (
                f"JSON:\n{json.dumps(gold_view, ensure_ascii=False)}\n\n"
                f"Текст:\n{text}"
            ),
        )

    def generate_gold_batch(self, specs: list[dict]) -> list[BankClientExtraction]:
        prompts = [self._gold_prompt(spec) for spec in specs]
        raw_list = self.gold_generator.batch(prompts, max_new_tokens=GEN_GOLD_MAX_NEW_TOKENS)
        profiles = self._parse_batch(raw_list, GoldProfileFill)
        merged = [
            merge_profile_and_prefill(profile, spec["prefill"])
            for profile, spec in zip(profiles, specs)
        ]
        return [
            apply_field_mask(model, spec["field_mask"])
            for model, spec in zip(merged, specs)
        ]

    def generate_text_batch(self, golds: list[BankClientExtraction]) -> list[str]:
        prompts = [self._client_text_prompt(gold) for gold in golds]
        raw_list = self.outlines_model.batch(prompts, max_new_tokens=GEN_TEXT_MAX_NEW_TOKENS)
        if not isinstance(raw_list, list):
            raw_list = [raw_list]
        return [str(raw).strip() for raw in raw_list]

    def check_coverage_batch(
        self,
        texts: list[str],
        golds: list[BankClientExtraction],
    ) -> list[CoverageCheck]:
        prompts = [
            self._coverage_prompt(text, gold)
            for text, gold in zip(texts, golds)
        ]
        raw_list = self.coverage_generator.batch(
            prompts,
            max_new_tokens=GEN_COVERAGE_MAX_NEW_TOKENS,
        )
        return self._parse_batch(raw_list, CoverageCheck)


def merge_profile_and_prefill(profile: GoldProfileFill, prefill: dict) -> BankClientExtraction:
    values = profile.model_dump()
    for key, value in prefill.items():
        if isinstance(value, BaseModel):
            values[key] = value.model_dump()
        else:
            values[key] = value
    return BankClientExtraction.model_validate(values)


def apply_field_mask(model: BankClientExtraction, field_mask: dict[str, bool]) -> BankClientExtraction:
    values = model.model_dump()
    for key, keep in field_mask.items():
        if not keep:
            values[key] = None
    return BankClientExtraction.model_validate(values)
