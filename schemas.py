from typing import List

from pydantic import BaseModel, Field


class CognitiveMistakes(BaseModel):
    name: str = Field(description="Когнитивная ошибка, выявленная в описании случая клиента.")
    explanation: str = Field(
        description="Краткое объяснение почему эта когнитивная ошибка применима к описанию случая клиента."
    )
    probability: float = Field(
        description="Вероятность того, что когнитивная ошибка верна для описания случая клиента.", ge=0.0, le=1.0
    )


class Attitude(BaseModel):
    content: str = Field(description="Мысленная установка к себе, людям и миру, взятая из описания случая клиента.")
    probability: float = Field(
        description="Вероятность того, что установка верна для описания случая клиента.", ge=0.0, le=1.0
    )


class RationalThough(BaseModel):
    content: str = Field(
        description="Рациональная мысль, корректирующая иррациональные мысли из описания случая клиента."
    )
    probability: float = Field(
        description="Вероятность того, что мысль верна для описания случая клиента.", ge=0.0, le=1.0
    )

class PositiveMoments(BaseModel):
    content: str = Field(
        description="Позитивные моменты в контексте описанного случая клиента."
    )
    probability: float = Field(
        description="Вероятность того, что момент верен для придания позитивного контекста случая клиента.", ge=0.0, le=1.0
    )


class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""

    summary: str = Field(description="Краткое резюме ответа, сгенерированного агентом")
    mistakes: List[CognitiveMistakes] = Field(
        default_factory=list,
        description="Список когнитивных ошибок, выявленных в описании случая клиента"
    )
    rationalization: List[RationalThough] = Field(
        default_factory=list,
        description="Список рациональных мыслей, корректирующих иррациональные мысли из описания случая клиента",
    )
    attitudes: List[Attitude] = Field(
        default_factory=list,
        description="Список мысленных установок к себе, людям и миру, взятых из описания случая клиента",
    )

    positivism: List[PositiveMoments] = Field(
        default_factory=list,
        description="Список позитивных моментов в контексте описанного случая клиента",
    )
