#!/usr/bin/env python3
"""Lists the public methods of the C++ API, one "<scope>\\t<name>" line each,
for compare.py, which checks that every one is exposed by a binding or
listed in cpp_exceptions.txt (issue #435).

  dump_cpp.py [--include DIR] <output file>

Simple parse of src/lib/include/libKriging/*.hpp: a method is a function
declaration of the public section of a class (exported or inline), or an
exported data member (std::function). Constructors,
destructors and operators are not methods. The members of KrigingImpl, base
class of Kriging, are listed in the Kriging scope. The free functions of the
headers are listed in the "libKriging" scope.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INCLUDE = os.path.join(HERE, "..", "..", "..", "src", "lib", "include", "libKriging")

# scope -> (header, class); the class members are listed in the scope
CLASSES = [
    ("Kriging", "Kriging.hpp", "Kriging"),
    ("Kriging", "KrigingImpl.hpp", "KrigingImpl"),
    ("WarpKriging", "WarpKriging.hpp", "WarpKriging"),
    ("MLPKriging", "MLPKriging.hpp", "MLPKriging"),
    ("NestedKriging", "NestedKriging.hpp", "NestedKriging"),
    ("Optim", "Optim.hpp", "Optim"),
    ("LinearAlgebra", "LinearAlgebra.hpp", "LinearAlgebra"),
    ("Random", "Random.hpp", "Random"),
    ("Covariance", "Covariance.hpp", "Covariance"),
    ("Trend", "Trend.hpp", "Trend"),
    ("KrigingLoader", "KrigingLoader.hpp", "KrigingLoader"),
]
# headers of free functions (scope "libKriging")
FREE_HEADERS = ["version.hpp"]
FREE_SCOPE = "libKriging"


def clean(text):
    """text without comments, preprocessor lines and the content of strings"""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r"^[ \t]*#[^\n]*(\\\n[^\n]*)*", " ", text, flags=re.M)
    text = re.sub(r'"(\\.|[^"\\])*"', '""', text)
    return re.sub(r"'(\\.|[^'\\])'", "' '", text)


def match_brace(text, start):
    """index of the brace closing the one at text[start]"""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def statements(body, default_access):
    """-> [(access, statement)] of the top level of a class body"""
    out, buf, access, i = [], [], default_access, 0
    while i < len(body):
        c = body[i]
        if c == ";":
            out.append((access, "".join(buf)))
            buf = []
        elif c == "{":
            end = match_brace(body, i)
            stmt = "".join(buf)
            # function definition: the statement ends with its body; type or
            # initializer: the statement goes on up to the ";"
            if "(" in re.sub(r"\[\[.*?\]\]", "", stmt) and not re.search(r"\b(struct|class|enum|union)\b", stmt) \
                    and not re.search(r"=\s*$", stmt):
                out.append((access, stmt))
                buf = []
            else:
                buf.append(" {} ")
            i = end
        elif c == ":" and body[i:i + 2] != "::" and body[i - 1:i + 1] != "::":
            label = "".join(buf).strip()
            if label in ("public", "private", "protected"):
                access, buf = label, []
            else:
                buf.append(c)
        else:
            buf.append(c)
        i += 1
    return out


def method_name(stmt, cls):
    """the name of the method declared by stmt, or None (not a method)"""
    s = re.sub(r"\[\[.*?\]\]", " ", stmt)
    s = s.replace("LIBKRIGING_EXPORT", " ")
    while True:  # template arguments, std::function<double()> included
        t = re.sub(r"<[^<>]*>", " ", s)
        if t == s:
            break
        s = t
    if re.search(r"\b(typedef|using|friend|enum)\b", s):
        return None
    m = re.search(r"(~?[A-Za-z_]\w*|operator\s*\S+?)\s*\(", s)
    if m is None:  # data member: only an exported one (std::function) is an entry point
        m = re.search(r"([A-Za-z_]\w*)\s*(=.*)?$", s.strip())
        return m.group(1) if m and "LIBKRIGING_EXPORT" in stmt else None
    name = m.group(1)
    if name.startswith(("~", "_", "operator")) or name == cls:
        return None
    return name


def class_methods(path, cls):
    with open(path) as f:
        text = clean(f.read())
    m = re.search(r"\b(class|struct)\s+(?:LIBKRIGING_EXPORT\s+)?%s\b[^;{]*\{" % re.escape(cls), text)
    if m is None:
        sys.exit("%s: class %s not found" % (path, cls))
    body = text[m.end():match_brace(text, m.end() - 1)]
    names = []
    for access, stmt in statements(body, "public" if m.group(1) == "struct" else "private"):
        if access != "public":
            continue
        name = method_name(stmt, cls)
        if name:
            names.append(name)
    return names


def free_functions(path):
    """the exported functions outside any class"""
    with open(path) as f:
        text = clean(f.read())
    names = []
    for stmt in text.split(";"):
        if "LIBKRIGING_EXPORT" in stmt and "(" in stmt and not re.search(r"\b(class|struct)\b", stmt):
            name = method_name(stmt, None)
            if name:
                names.append(name)
    return names


def dump(include):
    lines = set()
    for scope, header, cls in CLASSES:
        for name in class_methods(os.path.join(include, header), cls):
            lines.add((scope, name))
    for header in FREE_HEADERS:
        for name in free_functions(os.path.join(include, header)):
            lines.add((FREE_SCOPE, name))
    return sorted(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include", default=DEFAULT_INCLUDE, help="directory of the libKriging headers")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    with open(args.output, "w") as f:
        for scope, name in dump(args.include):
            f.write("%s\t%s\n" % (scope, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
