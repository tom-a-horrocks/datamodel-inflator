from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AusState(Enum):
    WA = "WA"
    SA = "SA"
    NT = "NT"
    TAS = "TAS"
    QLD = "QLD"
    VIC = "VIC"
    NSW = "NSW"
    ACT = "ACT"


@dataclass
class Street:
    apartment_no: Optional[int]
    number: int
    name: str


@dataclass
class Address:
    street: Street
    state: AusState


@dataclass
class Person:
    name: str
    aliases: list[str]
    delivery_address: Address
    billing_address: Optional[Address]
    gender: Optional[str]
