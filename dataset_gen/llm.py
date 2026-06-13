from __future__ import annotations

import json

import outlines
import torch
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ie_slm_bench.config import GENERATOR_MODEL, MAX_NEW_TOKENS
from ie_slm_bench.parsers import parse_outlines_output
from schemas.bank_client import Address, BankClientExtraction, WorkExperience


class PersonFill(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    surname: str = Field(alias="Фамилия")
    name: str = Field(alias="Имя")
    patronymic: str | None = Field(None, alias="Отчество")
    birth_place: str | None = Field(None, alias="Место рождения")
    citizenship: str | None = Field(None, alias="Гражданство")
    passport_issued_by: str | None = Field(None, alias="Кем выдан паспорт")
    employer: str | None = Field(None, alias="Место работы")
    job_title: str | None = Field(None, alias="Должность на работе")
    marital_status: str | None = Field(None, alias="Семейное положение")
    real_estate: str | None = Field(None, alias="Наличие недвижимости")
    car: str | None = Field(None, alias="Наличие автомобиля")
    registration_address: Address | None = Field(None, alias="Адрес регистрации")
    actual_address: Address | None = Field(None, alias="Адрес фактического проживания")


class ClientText(BaseModel):
    text: str


class CoverageCheck(BaseModel):
    all_present: bool
    missing_fields: list[str]


SKELETON_META_KEYS = {"sample_id", "person_mask", "det_mask", "_gender"}


class GeneratorBackend:
    def __init__(self, model_id: str = GENERATOR_MODEL, batch_size: int = 8):
        self.model_id = model_id
        self.batch_size = batch_size
        self.tokenizer = None
        self.hf_model = None
        self.outlines_model = None
        self.person_generator = None
        self.text_generator = None
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
        self.person_generator = outlines.Generator(self.outlines_model, PersonFill)
        self.text_generator = outlines.Generator(self.outlines_model, ClientText)
        self.coverage_generator = outlines.Generator(self.outlines_model, CoverageCheck)
        print(f"Generator {self.model_id}, batch_size={self.batch_size}, backend=outlines")

    def unload(self) -> None:
        del self.hf_model
        del self.tokenizer
        del self.outlines_model
        del self.person_generator
        del self.text_generator
        del self.coverage_generator
        self.hf_model = None
        self.tokenizer = None
        self.outlines_model = None
        self.person_generator = None
        self.text_generator = None
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

    def _person_fill_prompt(self, skeleton: dict) -> str:
        fixed_fields = {
            key: skeleton[key]
            for key in skeleton
            if key not in SKELETON_META_KEYS
        }
        return self._chat(
            "Ты заполняешь персональные поля клиента банка на русском языке. Верни только JSON по схеме.",
            (
                "Сгенерируй фамилию, имя, отчество и адреса с учётом пола.\n"
                f"Пол: {skeleton['_gender']}\n"
                f"Заполняй только поля с true в person_mask, остальные оставь null.\n"
                f"person_mask: {json.dumps(skeleton['person_mask'], ensure_ascii=False)}\n"
                f"Уже зафиксированные поля: {json.dumps(fixed_fields, ensure_ascii=False)}"
            ),
        )

    def _client_text_prompt(self, gold: BankClientExtraction) -> str:
        gold_view = gold.model_dump(by_alias=True, exclude_none=True)
        return self._chat(
            (
                "Ты пишешь фрагмент переписки клиента с сотрудником банка на русском языке. "
                "Текст должен содержать только те данные, которые указаны в JSON. "
                "Поля, отсутствующие в JSON, не упоминай и не выдумывай. "
                "Допустимо написать, что часть данных клиент укажет позже."
            ),
            f"JSON клиента:\n{json.dumps(gold_view, ensure_ascii=False, indent=2)}",
        )

    def _coverage_prompt(self, text: str, gold: BankClientExtraction) -> str:
        gold_view = gold.model_dump(by_alias=True, exclude_none=True)
        return self._chat(
            (
                "Проверь, что каждое непустое поле из JSON встречается в тексте клиента. "
                "Верни JSON: all_present=true, если потерь нет, иначе перечисли missing_fields."
            ),
            (
                f"JSON:\n{json.dumps(gold_view, ensure_ascii=False, indent=2)}\n\n"
                f"Текст:\n{text}"
            ),
        )

    def generate_person_fill_batch(self, skeletons: list[dict]) -> list[PersonFill]:
        prompts = [self._person_fill_prompt(skeleton) for skeleton in skeletons]
        raw_list = self.person_generator.batch(prompts, max_new_tokens=MAX_NEW_TOKENS)
        return self._parse_batch(raw_list, PersonFill)

    def generate_text_batch(self, golds: list[BankClientExtraction]) -> list[str]:
        prompts = [self._client_text_prompt(gold) for gold in golds]
        raw_list = self.text_generator.batch(prompts, max_new_tokens=MAX_NEW_TOKENS)
        client_texts = self._parse_batch(raw_list, ClientText)
        return [client_text.text for client_text in client_texts]

    def check_coverage_batch(
        self,
        texts: list[str],
        golds: list[BankClientExtraction],
    ) -> list[CoverageCheck]:
        prompts = [
            self._coverage_prompt(text, gold)
            for text, gold in zip(texts, golds)
        ]
        raw_list = self.coverage_generator.batch(prompts, max_new_tokens=256)
        return self._parse_batch(raw_list, CoverageCheck)


def merge_skeleton_and_person(skeleton: dict, person: PersonFill) -> BankClientExtraction:
    work_experience = None
    if skeleton["work_experience_years"] is not None or skeleton["work_experience_months"] is not None:
        work_experience = WorkExperience(
            years=skeleton["work_experience_years"],
            months=skeleton["work_experience_months"],
        )
    person_mask = skeleton["person_mask"]
    return BankClientExtraction(
        surname=person.surname,
        name=person.name,
        patronymic=person.patronymic,
        birth_date=skeleton["birth_date"],
        birth_year=skeleton["birth_year"],
        birth_place=person.birth_place if person_mask["birth_place"] else None,
        citizenship=person.citizenship if person_mask["citizenship"] else None,
        gender=skeleton["gender"],
        passport_series_number=skeleton["passport_series_number"],
        passport_issued_by=person.passport_issued_by if person_mask["passport_issued_by"] else None,
        passport_issue_date=skeleton["passport_issue_date"],
        passport_department_code=skeleton["passport_department_code"],
        inn=skeleton["inn"],
        snils=skeleton["snils"],
        registration_address=person.registration_address if person_mask["registration_address"] else None,
        actual_address=person.actual_address if person_mask["actual_address"] else None,
        mobile_phone=skeleton["mobile_phone"],
        email=skeleton["email"],
        employer=person.employer if person_mask["employer"] else None,
        job_title=person.job_title if person_mask["job_title"] else None,
        work_experience=work_experience,
        monthly_income=skeleton["monthly_income"],
        marital_status=person.marital_status if person_mask["marital_status"] else None,
        dependents_count=skeleton["dependents_count"],
        real_estate=person.real_estate if person_mask["real_estate"] else None,
        car=person.car if person_mask["car"] else None,
        loans_count=skeleton["loans_count"],
    )
