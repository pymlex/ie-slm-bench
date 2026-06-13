from pydantic import BaseModel, ConfigDict, Field


class Address(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    index: str | None = Field(None, alias="индекс")
    country: str | None = Field(None, alias="страна")
    region: str | None = Field(None, alias="регион")
    city: str | None = Field(None, alias="город")
    street: str | None = Field(None, alias="улица")
    house: str | None = Field(None, alias="дом")
    apartment: str | None = Field(None, alias="квартира")


class WorkExperience(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    years: int | None = Field(None, alias="лет")
    months: int | None = Field(None, alias="месяцев")


class BankClientExtraction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    surname: str | None = Field(None, alias="Фамилия")
    name: str | None = Field(None, alias="Имя")
    patronymic: str | None = Field(None, alias="Отчество")
    birth_date: str | None = Field(None, alias="Дата рождения")
    birth_year: int | None = Field(None, alias="Год рождения")
    birth_place: str | None = Field(None, alias="Место рождения")
    citizenship: str | None = Field(None, alias="Гражданство")
    gender: str | None = Field(None, alias="Пол")
    passport_series_number: str | None = Field(None, alias="Серия и номер паспорта")
    passport_issued_by: str | None = Field(None, alias="Кем выдан паспорт")
    passport_issue_date: str | None = Field(None, alias="Дата выдачи паспорта")
    passport_department_code: str | None = Field(None, alias="Код подразделения")
    inn: str | None = Field(None, alias="ИНН")
    snils: str | None = Field(None, alias="СНИЛС")
    registration_address: Address | None = Field(None, alias="Адрес регистрации")
    actual_address: Address | None = Field(None, alias="Адрес фактического проживания")
    mobile_phone: str | None = Field(None, alias="Номер мобильного телефона")
    email: str | None = Field(None, alias="Адрес электронной почты")
    employer: str | None = Field(None, alias="Место работы")
    job_title: str | None = Field(None, alias="Должность на работе")
    work_experience: WorkExperience | None = Field(None, alias="Стаж работы")
    monthly_income: str | None = Field(None, alias="Ежемесячный доход")
    marital_status: str | None = Field(None, alias="Семейное положение")
    dependents_count: int | None = Field(None, alias="Количество иждивенцев")
    real_estate: str | None = Field(None, alias="Наличие недвижимости")
    car: str | None = Field(None, alias="Наличие автомобиля")
    loans_count: int | None = Field(None, alias="Наличие кредитов/займов")


class GoldProfileFill(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    surname: str = Field(alias="Фамилия")
    name: str = Field(alias="Имя")
    patronymic: str | None = Field(None, alias="Отчество")
    birth_place: str | None = Field(None, alias="Место рождения")
    citizenship: str | None = Field(None, alias="Гражданство")
    passport_issued_by: str | None = Field(None, alias="Кем выдан паспорт")
    registration_address: Address | None = Field(None, alias="Адрес регистрации")
    actual_address: Address | None = Field(None, alias="Адрес фактического проживания")
    employer: str | None = Field(None, alias="Место работы")
    job_title: str | None = Field(None, alias="Должность на работе")
    marital_status: str | None = Field(None, alias="Семейное положение")
    real_estate: str | None = Field(None, alias="Наличие недвижимости")
    car: str | None = Field(None, alias="Наличие автомобиля")
