import dataclasses
from collections.abc import Iterable, Callable
from enum import Enum
from itertools import chain
from operator import itemgetter
from typing import (
    get_origin,
    Union,
    get_args,
    TypeVar,
    get_type_hints,
    Any,
    Type,
)

from black import format_str, FileMode

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
K = TypeVar("K")
X = TypeVar("X")
Y = TypeVar("Y")
T = TypeVar("T")
V = TypeVar("V")


def identity(x: T) -> T:
    return x


def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def fog(x: A) -> C:
        return f(g(x))

    return fog


def field(name: str, f: Callable[[X], Y]) -> Callable[[dict[str, Any]], Y]:
    return compose(f, itemgetter(name))


def field_allow_missing(
    name: str, f: Callable[[X | None], Y]
) -> Callable[[dict[str, Any]], Y]:
    """
    Like field, but if "name" not in dict then None is passed to f.
    """

    def get_or_none(k: str) -> Callable[[dict[str, X]], X | None]:
        return lambda d: d.get(k)

    return compose(f, get_or_none(name))


def obj_parser(klass: Type[Y], **parsers) -> Callable[[dict[str, Any]], Y]:
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


def make_list_parser(inner_tp: Type[T]) -> Callable[[list[Any]], list[T]]:
    inner_parser = make_parser(inner_tp)
    return lambda xs: [inner_parser(x) for x in xs]


def make_optional_parser(inner_tp: Type[T | None]) -> Callable[[Any], T | None]:
    inner_parser = make_parser(inner_tp)
    return lambda x: None if x is None else inner_parser(x)


def make_dict_parser(
    key_type: Type[K], value_type: Type[V]
) -> Callable[[dict[Any, Any]], dict[K, V]]:
    key_parser = make_parser(key_type)
    value_parser = make_parser(value_type)
    return lambda d: {key_parser(k): value_parser(v) for k, v in d.items()}


def make_parser(tp: Type[T]) -> Callable[[Any], T]:
    # Argument could be a (data)class, a wrapped type (list or optional), or a base class (inc. Enum).
    if origin := get_origin(tp):
        # LIST or OPTIONAL
        inner_types = get_args(tp)
        if origin is list:
            # TODO fix type error
            return make_list_parser(only(inner_types))  # type: ignore[return-value]
        elif origin is Union and len(inner_types) == 2 and inner_types[1] is type(None):
            # TODO fix type error
            return make_optional_parser(inner_types[0])  # type: ignore[return-value]
        elif origin is dict:
            k_type, v_type = inner_types
            # TODO fix type error
            return make_dict_parser(k_type, v_type)  # type: ignore[return-value]
        else:
            raise ValueError(
                f"Unexpected compound type (only Union, List, and Dict supported): {tp}"
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
            f.name: field(name=f.name, f=make_parser(type_hints[f.name]))
            if f.default == dataclasses.MISSING
            else field_allow_missing(name=f.name, f=make_parser(type_hints[f.name]))
            for f in fields
        }
        # TODO fix type error
        return obj_parser(tp, **parser_dict)  # type: ignore[arg-type]
    elif issubclass(tp, Enum):
        # Assume enum class is its own constructor
        return tp  # type: ignore[return-value]
    elif tp in {str, int, float, bool}:
        # Already parsed
        return identity
    raise ValueError(f"Unsupported type {tp}")


def make_list_parser_code(inner_tp: Type[T], dc_parsers_exist: bool) -> str:
    inner_parser_code = make_parser_statement(
        inner_tp, dc_parsers_exist, _at_root=False
    )
    return f"lambda xs: [{inner_parser_code}(x) for x in xs]"


def make_dict_parser_code(
    key_type: Type[K], value_type: Type[V], dc_parsers_exist: bool
) -> str:
    key_parser_code = make_parser_statement(key_type, dc_parsers_exist, _at_root=False)
    value_parser_code = make_parser_statement(
        value_type, dc_parsers_exist, _at_root=False
    )
    return f"lambda d: {{{key_parser_code}(k): {value_parser_code}(v) for k,v in d.items()}}"


def make_optional_parser_code(inner_tp: Type[T | None], dc_parsers_exist: bool) -> str:
    inner_parser_code = make_parser_statement(
        inner_tp, dc_parsers_exist, _at_root=False
    )
    return f"lambda x: None if x is None else ({inner_parser_code})(x)"


def dc_parser_name(dc: Type) -> str:
    return f"{dc.__name__.lower()}_parser"


def make_parser_statement(tp: Type[T], recurse_dc: bool, _at_root: bool = True) -> str:
    # WRAPPED TYPE (list, origin)
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is list:
        return make_list_parser_code(only(args), recurse_dc)
    if origin is Union and len(args) == 2 and args[1] is type(None):
        return make_optional_parser_code(args[0], recurse_dc)
    if origin is dict:
        key_type, value_type = args
        return make_dict_parser_code(key_type, value_type, recurse_dc)
    if origin is not None:
        raise ValueError(
            f"Unexpected compound type (only Union, List, and Dict supported): {tp}"
        )
    # DATACLASS
    if dataclasses.is_dataclass(tp):
        if recurse_dc or _at_root:
            fields = dataclasses.fields(tp)
            type_hints = get_type_hints(tp)
            return (
                f"obj_parser({tp.__name__},"
                + ",".join(
                    f"{f.name}=field("
                    f"name='{f.name}', "
                    f"f={make_parser_statement(tp=type_hints[f.name], recurse_dc=recurse_dc, _at_root=False)}, "
                    f"has_default={f.default != dataclasses.MISSING}"
                    f")"
                    for f in fields
                )
                + ")"
            )
        # Otherwise assume parser exists
        return dc_parser_name(tp)

    # ENUM
    if issubclass(tp, Enum):
        return tp.__name__
    # BASE
    if tp in {str, int, float, bool, Any}:
        return "identity"
    raise ValueError(f"Unsupported type {tp}")


def traverse_dataclasses(tp: Type) -> list:
    """
    Construct a complete list of all dataclasses referenced in the given dataclass,
    recursively, and including the given dataclass itself. If the given type is not
    a dataclass an empty list is returned.

    The discovered types are ordered from the parent to the leaves, and may contain duplicates.
    :param tp:
    :return:
    """
    # WRAPPED TYPE (list, origin)
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is list:
        return traverse_dataclasses(only(args))
    if origin is Union and len(args) == 2 and args[1] is type(None):
        return traverse_dataclasses(args[0])
    if origin is dict:
        key_type, value_type = args
        return traverse_dataclasses(key_type) + traverse_dataclasses(value_type)
    if origin is not None:
        raise ValueError(
            f"Unexpected compound type (only Union, List, and Dict supported): {tp}"
        )
    if dataclasses.is_dataclass(tp):
        # DATACLASS
        type_hints = get_type_hints(tp)
        field_types = [type_hints[f.name] for f in dataclasses.fields(tp)]
        return list(
            chain([tp], *[traverse_dataclasses(inner_tp) for inner_tp in field_types])
        )
    return []


def filter_unique(xs: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(xs))


def dataclass_dependencies(dc: Type) -> list:
    return filter_unique(reversed(traverse_dataclasses(dc)))


def generate_parser_code(dcs: list[Type]) -> str:
    dcs = filter_unique(chain.from_iterable(dataclass_dependencies(dc) for dc in dcs))
    import_statements = ["from parser_generator import *", "from model import *"]
    parser_statements = [
        f"{dc_parser_name(dc)} = {make_parser_statement(dc, recurse_dc=False)}"
        for dc in dcs
    ]
    code = "\n".join(import_statements + parser_statements)
    formatted_code = format_str(code, mode=FileMode())
    return formatted_code
