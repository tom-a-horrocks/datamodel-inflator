import dataclasses
from collections.abc import Iterable, Callable
from enum import Enum
from operator import itemgetter
from typing import (
    get_origin,
    Union,
    get_args,
    TypeVar,
    get_type_hints,
    Any,
    Optional,
)


A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
X = TypeVar("X")
Y = TypeVar("Y")
T = TypeVar("T")


def identity(x: T) -> T:
    return x


def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def fog(x: A) -> C:
        return f(g(x))

    return fog


def field(
    name: str, f: Callable[[X], Y], has_default=False
) -> Callable[[dict[str, Any]], Y]:
    def get_or_none(k: str) -> Callable[[dict[str, X]], Optional[X]]:
        return lambda d: d.get(k)

    get = get_or_none(name) if has_default else itemgetter(name)
    return compose(f, get)


def obj_parser(klass: type[Y], **parsers) -> Callable[[dict[str, Any]], Y]:
    def inner(d: dict[str, Any]) -> Y:
        return klass(**{attr: parser(d) for attr, parser in parsers.items()})

    return inner


def only(xs: Iterable[T]) -> T:
    it = iter(xs)
    fst = next(it)
    try:
        snd = next(it)
        raise ValueError("Should be single-element iterable, but was", [fst, snd, *it])
    except StopIteration:
        pass
    return fst


def make_list_parser(inner_tp: type[T]) -> Callable[[list[Any]], T]:
    inner_parser = make_parser(inner_tp)
    return lambda xs: [inner_parser(x) for x in xs]


def make_optional_parser(inner_tp: type[Optional[T]]) -> Callable[[Any], Optional[T]]:
    inner_parser = make_parser(inner_tp)
    return lambda x: None if x is None else inner_parser(x)


def make_parser(tp: type[T]) -> Callable[[Any], T]:
    # Argument could be a (data)class, a wrapped type (list or optional), or a base class (inc. Enum).
    if origin := get_origin(tp):
        # LIST or OPTIONAL
        args = get_args(tp)
        if origin is list:
            return make_list_parser(only(args))
        elif origin is Union and len(args) == 2 and args[1] is type(None):
            return make_optional_parser(args[0])
        else:
            raise ValueError(
                f"Unexpected compound type (only Union and List supported): {tp}"
            )
        # Wrapped type (list or optional)
    elif dataclasses.is_dataclass(tp):
        # DATACLASS
        fields = dataclasses.fields(tp)
        type_hints = get_type_hints(tp)
        # The type *must* come from get_type_hints and NOT from dataclasses.field.type,
        # because `from __future__ import annotations` in model.py changes the
        # dataclasses.field.type from actual types (e.g. int) to a string representations
        # of that type (e.g. "int").
        parser_dict = {
            f.name: field(
                name=f.name,
                f=make_parser(type_hints[f.name]),
                has_default=f.default != dataclasses.MISSING,
            )
            for f in fields
        }
        return obj_parser(tp, **parser_dict)
    elif issubclass(tp, Enum):
        return tp  # Assume enum class is its own constructor
    elif tp in {dict, str, int, float, bool}:
        # Already parsed
        return identity
    raise ValueError(f"Unsupported type {tp}")


def make_list_parser_code(inner_tp: type[T]) -> tuple[str, str]:
    inner_parser_code = make_parser_code(inner_tp)
    return f"lambda xs: [{inner_parser_code}(x) for x in xs]"


def make_optional_parser_code(
    inner_tp: type[Optional[T]],
) -> tuple[str, str]:
    inner_parser_code = make_parser_code(inner_tp)
    return f"lambda x: None if x is None else {inner_parser_code}(x)"


def make_parser_code(tp: type[T]) -> tuple[str, str]:
    # Argument could be a (data)class, a wrapped type (list or optional), or a base class (inc. Enum).
    if origin := get_origin(tp):
        # LIST or OPTIONAL
        args = get_args(tp)
        if origin is list:
            return make_list_parser_code(only(args))
        elif origin is Union and len(args) == 2 and args[1] is type(None):
            return make_optional_parser_code(args[0])
        else:
            raise ValueError(
                f"Unexpected compound type (only Union and List supported): {tp}"
            )
        # Wrapped type (list or optional)
    elif dataclasses.is_dataclass(tp):
        # DATACLASS
        fields = dataclasses.fields(tp)
        type_hints = get_type_hints(tp)

        # First, generate parsers for each dataclass field. This is first so that it is defined before it's needed.
        retval = f"{tp.__name__}_parser = obj_parser({tp.__name__},"
        for f in fields:
            defn, body = make_parser_code(type_hints[f.name])
            retval += f"{f.name}=field(name='{f.name}', f={make_parser_code(type_hints[f.name])}, has_default={f.default != dataclasses.MISSING})"
        retval += ")"

        # Second, return the parser for the current dataclass field.

        return (
            f"obj_parser({tp.__name__},"
            + ",".join(
                f"{f.name}=field(name='{f.name}', f={make_parser_code(type_hints[f.name])}, has_default={f.default != dataclasses.MISSING})"
                for f in fields
            )
            + ")"
        )

    elif issubclass(tp, Enum):
        return tp.__name__
    elif tp in {dict, str, int, float, bool}:
        return "identity"
    raise ValueError(f"Unsupported type {tp}")


def test():
    from dataclasses import dataclass
    from black import format_str, FileMode

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
    class Address:
        number: int
        street: str
        state: AusState

    @dataclass
    class Person:
        name: str
        aliases: list[str]
        address: Address
        gender: Optional[str]

    address = Address(13, "Fake Street", AusState.WA)
    person = Person("Tom", ["T-bone", "t3h pwn3r3r"], address, None)

    code = make_parser_code(Person)

    print(code)
    print(format_str(code, mode=FileMode()))


if __name__ == "__main__":
    test()
