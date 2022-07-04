import textwrap

from . import defs
from typing import Sequence, Final, Type
import jinja2
import numpy
import ctypes


language_id = "cpp"


def render_python_type(python_type: Type):
    mapping: Final = {
        bool: "bool",
        int: "long",
        float: "double",
        complex: "std::complex<double>",
        numpy.bool_: "bool",
        numpy.byte: "signed char",
        numpy.ubyte: "unsigned char",
        numpy.short: "short",
        numpy.ushort: "unsigned short",
        numpy.intc: "int",
        numpy.uintc: "unsigned int",
        numpy.int_: "long",
        numpy.uint: "unsigned long",
        numpy.longlong: "long long",
        numpy.ulonglong: "unsigned long long",
        numpy.single: "float",
        numpy.double: "double",
        numpy.longdouble: "long double",
        numpy.csingle: "std::complex<float>",
        numpy.cdouble: "std::complex<double>",
        numpy.clongdouble: "std::complex<long double>",
        ctypes.c_bool: "bool",
        ctypes.c_char: "char",
        ctypes.c_wchar: "wchar_t",
        ctypes.c_byte: "char",
        ctypes.c_ubyte: "unsigned char",
        ctypes.c_short: "short",
        ctypes.c_ushort: "unsigned short",
        ctypes.c_int: "int",
        ctypes.c_uint: "unsigned int",
        ctypes.c_long: "long",
        ctypes.c_ulong: "unsigned long",
        ctypes.c_longlong: "long long",
        ctypes.c_ulonglong: "unsigned long long",
        ctypes.c_size_t: "std::size_t",
        ctypes.c_ssize_t: "std::ptrdiff_t",
        ctypes.c_float: "float",
        ctypes.c_double: "double",
        ctypes.c_longdouble: "long double",
    }
    return mapping[python_type]


def _render_function_param(param: [defs.ScalarParameter | defs. BufferParameter], index: int):
    if isinstance(param, defs.ScalarParameter):
        return "{type} {name}".format(type=render_python_type(param.type_), name=param.name)
    else:
        template_type = "BufferT{index}&&".format(index=index)
        return "{type} {name}".format(type=template_type, name=param.name)


def render_function_declaration(function: defs.Function, body: str) -> str:
    decl_templ = jinja2.Template(textwrap.dedent("""\
    decltype(auto) {{name}}({{", ".join(parameters)}}) {
        {{body}}
    }\
    """))
    rendered_params = [_render_function_param(param, index) for index, param in enumerate(function.parameters)]
    rendered_decl = decl_templ.render(name=function.name, parameters=rendered_params, body=body)
    template_params = ["class BufferT{index}".format(index=index)
                       for index, param in enumerate(function.parameters)
                       if isinstance(param, defs.BufferParameter)]
    if template_params:
        render_tpl = jinja2.Template("""template <{{", ".join(template_params)}}>""")
        rendered_tpl_params = render_tpl.render(template_params=template_params)
        rendered_decl = rendered_tpl_params + "\n" + rendered_decl
    return rendered_decl


def render_function_call(function: defs.Function, args: Sequence[str]) -> str:
    call_templ = jinja2.Template("""{{name}}({{", ".join(args)}})""")
    return call_templ.render(name=function.name, args=args)