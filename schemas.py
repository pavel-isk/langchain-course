from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Рецензия о том чего не хватает.")
    suporfluous: str = Field(description="Рецензия о том что излишнее.")


class AnswerQuestion(BaseModel):
    """Ответ на вопрос."""

    answer: str = Field(description="Детальный ответ на вопрос, приблизительно 250 слов.")
    reflection: Reflection = Field(description="Твои обдуманные мысли по поводу изначального ответа.")
    search_queries: list[str] = Field(
        description="1-3 поисковых запроса для поиска информации чтобы чтоб улучшить ответ в соответствии с рецензией."
    )


class ReviseAnswer(AnswerQuestion):
    references: list[str] = Field(description="список цитат, который повлиял на твоей обновлённый ответ.")
