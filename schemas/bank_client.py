from pydantic import BaseModel, ConfigDict, Field, field_validator


class Address(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    index: str | None = Field(None, alias="индекс", max_length=12)
    country: str | None = Field(None, alias="страна", max_length=80)
    region: str | None = Field(None, alias="регион", max_length=120)
    city: str | None = Field(None, alias="город", max_length=80)
    street: str | None = Field(None, alias="улица", max_length=120)
    house: str | None = Field(None, alias="дом", max_length=32)
    apartment: str | None = Field(None, alias="квартира", max_length=32)


class WorkExperience(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    years: int | None = Field(None, alias="лет")
    months: int | None = Field(None, alias="месяцев")


class BankClientExtraction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    surname: str | None = Field(None, alias="Фамилия", max_length=80)
    name: str | None = Field(None, alias="Имя", max_length=80)
    patronymic: str | None = Field(None, alias="Отчество", max_length=80)
    birth_date: str | None = Field(None, alias="Дата рождения", max_length=16)
    birth_year: int | None = Field(None, alias="Год рождения")
    birth_place: str | None = Field(None, alias="Место рождения", max_length=160)
    citizenship: str | None = Field(None, alias="Гражданство", max_length=80)
    gender: str | None = Field(None, alias="Пол", max_length=8)
    passport_series_number: str | None = Field(None, alias="Серия и номер паспорта", max_length=16)
    passport_issued_by: str | None = Field(None, alias="Кем выдан паспорт", max_length=240)
    passport_issue_date: str | None = Field(None, alias="Дата выдачи паспорта", max_length=16)
    passport_department_code: str | None = Field(None, alias="Код подразделения", max_length=16)
    inn: int | None = Field(None, alias="ИНН", ge=0, le=999999999999)
    snils: str | None = Field(None, alias="СНИЛС", max_length=14)
    registration_address: Address | None = Field(None, alias="Адрес регистрации")
    actual_address: Address | None = Field(None, alias="Адрес фактического проживания")
    mobile_phone: str | None = Field(None, alias="Номер мобильного телефона", max_length=20)
    email: str | None = Field(None, alias="Адрес электронной почты", max_length=120)
    employer: str | None = Field(None, alias="Место работы", max_length=160)
    job_title: str | None = Field(None, alias="Должность на работе", max_length=120)
    work_experience: WorkExperience | None = Field(None, alias="Стаж работы")
    monthly_income: int | None = Field(None, alias="Ежемесячный доход", ge=0, le=999999999)
    marital_status: str | None = Field(None, alias="Семейное положение", max_length=80)
    dependents_count: int | None = Field(None, alias="Количество иждивенцев", ge=0, le=20)
    real_estate: str | None = Field(None, alias="Наличие недвижимости", max_length=80)
    car: str | None = Field(None, alias="Наличие автомобиля", max_length=80)
    loans_count: int | None = Field(None, alias="Наличие кредитов/займов", ge=0, le=50)

    @field_validator("inn", "monthly_income", mode="before")
    @classmethod
    def coerce_int_fields(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            return int(stripped)
        return value


class GoldAddressFill(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    city: str = Field(alias="город")
    street: str = Field(alias="улица")
    house: str = Field(alias="дом")
    index: str | None = Field(None, alias="индекс")
    country: str | None = Field(None, alias="страна")
    region: str | None = Field(None, alias="регион")
    apartment: str | None = Field(None, alias="квартира")


class GoldProfileFill(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    surname: str = Field(alias="Фамилия")
    name: str = Field(alias="Имя")
    patronymic: str | None = Field(None, alias="Отчество")
    birth_place: str = Field(alias="Место рождения")
    citizenship: str = Field(alias="Гражданство")
    passport_issued_by: str = Field(alias="Кем выдан паспорт")
    registration_address: GoldAddressFill = Field(alias="Адрес регистрации")
    actual_address: GoldAddressFill = Field(alias="Адрес фактического проживания")
    employer: str = Field(alias="Место работы")
    job_title: str = Field(alias="Должность на работе")
    marital_status: str = Field(alias="Семейное положение")
    real_estate: str = Field(alias="Наличие недвижимости")
    car: str = Field(alias="Наличие автомобиля")
