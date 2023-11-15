from parser_generator import *
from model import *

street_parser = obj_parser(
    Street,
    apartment_no=field(
        name="apartment_no",
        f=lambda x: None if x is None else identity(x),
        has_default=False,
    ),
    number=field(name="number", f=identity, has_default=False),
    name=field(name="name", f=identity, has_default=False),
)
address_parser = obj_parser(
    Address,
    street=field(name="street", f=street_parser, has_default=False),
    state=field(name="state", f=AusState, has_default=False),
)
person_parser = obj_parser(
    Person,
    name=field(name="name", f=identity, has_default=False),
    aliases=field(
        name="aliases", f=lambda xs: [identity(x) for x in xs], has_default=False
    ),
    delivery_address=field(
        name="delivery_address", f=address_parser, has_default=False
    ),
    billing_address=field(
        name="billing_address",
        f=lambda x: None if x is None else address_parser(x),
        has_default=False,
    ),
    gender=field(
        name="gender", f=lambda x: None if x is None else identity(x), has_default=False
    ),
)
