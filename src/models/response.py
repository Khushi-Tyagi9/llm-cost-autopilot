from dataclasses import dataclass


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float
    cost: float
    model_id: str
    provider: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        return (
            f"[{self.provider}/{self.model_id}] "
            f"{self.total_tokens} tok | {self.latency:.2f}s | ${self.cost:.6f}"
        )