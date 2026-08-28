from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    decision: Literal["same", "different", "review"]
    actor: str = Field(default="operator", min_length=1, max_length=80)


class PurgeRequest(BaseModel):
    confirmation: str
    actor: str = Field(default="operator", min_length=1, max_length=80)


class ReplayControlRequest(BaseModel):
    action: Literal["play", "pause", "seek", "speed"]
    position: float | None = Field(default=None, ge=0, le=1)
    speed: int | None = Field(default=None, ge=1, le=64)


class CalibrationRequest(BaseModel):
    camera_id: str
    approach_zone: list[tuple[float, float]]
    commit_line: tuple[tuple[float, float], tuple[float, float]]
    direction: Literal["in", "out"] = "in"
