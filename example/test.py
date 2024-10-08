from pathlib import Path

from ..parser_generator import generate_parser_code, make_parser
from .model import Address, AusState, Person, Street

person = Person(
    name="Tom",
    aliases=["T-bone", "t3h pwn3r3r"],
    delivery_address=Address(Street(None, 13, "Fake Street"), AusState.WA),
    billing_address=None,
    gender=None,
)

person_dict = {
    "name": "Tom",
    "aliases": ["T-bone", "t3h pwn3r3r"],
    "delivery_address": {
        "street": {"apartment_no": None, "number": 13, "name": "Fake Street"},
        "state": "WA",
    },
    "billing_address": None,
    "gender": None,
}

# Test dynamic parsing
person_parser = make_parser(Person)
assert person_parser(person_dict) == person

# Test static parsing
formatted_code = generate_parser_code([Person])
Path("./parser.py").write_text(formatted_code)

from . import parser

assert parser.person_parser(person_dict) == person
