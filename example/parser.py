from ..parser_generator import field, identity, obj_parser
from .model import Address, AusState, Person, Street

street_parser = obj_parser(
    Street,
    apartment_no=field(
        name="apartment_no", f=lambda x: None if x is None else (identity)(x)
    ),
    number=field(name="number", f=identity),
    name=field(name="name", f=identity),
)
address_parser = obj_parser(
    Address,
    street=field(name="street", f=street_parser),
    state=field(name="state", f=AusState),
)
person_parser = obj_parser(
    Person,
    name=field(name="name", f=identity),
    aliases=field(name="aliases", f=lambda xs: [identity(x) for x in xs]),  # type: ignore
    delivery_address=field(name="delivery_address", f=address_parser),
    billing_address=field(
        name="billing_address",
        f=lambda x: None if x is None else (address_parser)(x),  # type: ignore
    ),
    gender=field(name="gender", f=lambda x: None if x is None else (identity)(x)),
)
