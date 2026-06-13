from __future__ import annotations

import json


ADDRESS_TEMPLATE = {
    "индекс": "verbatim-string",
    "страна": "verbatim-string",
    "регион": "verbatim-string",
    "город": "verbatim-string",
    "улица": "verbatim-string",
    "дом": "verbatim-string",
    "квартира": "verbatim-string",
}

WORK_EXPERIENCE_TEMPLATE = {
    "лет": "integer",
    "месяцев": "integer",
}

BANK_CLIENT_NUEXTRACT_TEMPLATE = json.dumps(
    {
        "Фамилия": "verbatim-string",
        "Имя": "verbatim-string",
        "Отчество": "verbatim-string",
        "Дата рождения": "verbatim-string",
        "Год рождения": "integer",
        "Место рождения": "verbatim-string",
        "Гражданство": "verbatim-string",
        "Пол": "verbatim-string",
        "Серия и номер паспорта": "verbatim-string",
        "Кем выдан паспорт": "verbatim-string",
        "Дата выдачи паспорта": "verbatim-string",
        "Код подразделения": "verbatim-string",
        "ИНН": "integer",
        "СНИЛС": "verbatim-string",
        "Адрес регистрации": ADDRESS_TEMPLATE,
        "Адрес фактического проживания": ADDRESS_TEMPLATE,
        "Номер мобильного телефона": "verbatim-string",
        "Адрес электронной почты": "verbatim-string",
        "Место работы": "verbatim-string",
        "Должность на работе": "verbatim-string",
        "Стаж работы": WORK_EXPERIENCE_TEMPLATE,
        "Ежемесячный доход": "integer",
        "Семейное положение": "verbatim-string",
        "Количество иждивенцев": "integer",
        "Наличие недвижимости": "verbatim-string",
        "Наличие автомобиля": "verbatim-string",
        "Наличие кредитов/займов": "integer",
    },
    ensure_ascii=False,
)
