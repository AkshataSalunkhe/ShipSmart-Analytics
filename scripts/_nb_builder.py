"""
Turn a percent-format (# %%) annotated .py script into a real, pre-executed
.ipynb notebook -- without depending on jupyter/nbformat/nbclient (not
installable in this environment). We execute each cell in a persistent
namespace exactly like a kernel would, capture stdout/stderr, capture any
matplotlib figures created during the cell as embedded PNGs, and capture
the repr of a trailing bare expression (the way Jupyter auto-displays the
last line of a cell). The result is a standard nbformat-v4 JSON file that
opens in Jupyter/VS Code/nbviewer with all outputs already rendered.
"""
import ast
import base64
import io
import json
import sys
import contextlib
import warnings


def parse_cells(source: str):
    lines = source.split("\n")
    cells = []
    cur_type, cur_lines = None, []
    for line in lines:
        if line.startswith("# %%"):
            if cur_type is not None:
                cells.append((cur_type, cur_lines))
            cur_type = "markdown" if "[markdown]" in line else "code"
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_type is not None:
        cells.append((cur_type, cur_lines))
    return cells


def markdown_source(lines):
    out = []
    for l in lines:
        if l.startswith("# "):
            out.append(l[2:])
        elif l.strip() == "#":
            out.append("")
        else:
            out.append(l)
    text = "\n".join(out).strip("\n")
    return text


def code_source(lines):
    text = "\n".join(lines).strip("\n")
    return text


def split_source_lines(text):
    if text == "":
        return []
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def run_cell(source, ns, filename):
    try:
        tree = ast.parse(source, filename=filename, mode="exec")
    except SyntaxError:
        exec(compile(source, filename, "exec"), ns)
        return "", "", None
    body = tree.body
    trailing_expr = None
    if body and isinstance(body[-1], ast.Expr):
        trailing_expr = body.pop()

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    result = None
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if body:
                mod = ast.Module(body=body, type_ignores=[])
                exec(compile(mod, filename, "exec"), ns)
            if trailing_expr is not None:
                expr = ast.Expression(body=trailing_expr.value)
                result = eval(compile(expr, filename, "eval"), ns)
    return stdout_buf.getvalue(), stderr_buf.getvalue(), result


def build_notebook(py_path, ipynb_path, exec_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(py_path) as f:
        source = f.read()
    cells_raw = parse_cells(source)

    ns = {"__name__": "__main__"}
    if exec_prefix:
        exec(compile(exec_prefix, "<prefix>", "exec"), ns)

    nb_cells = []
    exec_count = 0

    for cell_type, lines in cells_raw:
        if cell_type == "markdown":
            text = markdown_source(lines)
            if text.strip() == "":
                continue
            nb_cells.append({
                "cell_type": "markdown", "metadata": {}, "source": split_source_lines(text),
            })
        else:
            text = code_source(lines)
            if text.strip() == "":
                continue
            exec_count += 1
            before_figs = set(plt.get_fignums())
            stdout, stderr, result = run_cell(text, ns, filename=f"{py_path}:cell{exec_count}")
            after_figs = [n for n in plt.get_fignums() if n not in before_figs]

            outputs = []
            if stdout:
                outputs.append({"output_type": "stream", "name": "stdout", "text": split_source_lines(stdout.rstrip("\n"))})
            if stderr:
                outputs.append({"output_type": "stream", "name": "stderr", "text": split_source_lines(stderr.rstrip("\n"))})
            if result is not None:
                data = {"text/plain": split_source_lines(repr(result))}
                if hasattr(result, "_repr_html_"):
                    try:
                        html = result._repr_html_()
                        if html:
                            data["text/html"] = split_source_lines(html)
                    except Exception:
                        pass
                outputs.append({
                    "output_type": "execute_result", "execution_count": exec_count,
                    "data": data, "metadata": {},
                })
            for fignum in after_figs:
                fig = plt.figure(fignum)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("ascii")
                outputs.append({
                    "output_type": "display_data",
                    "data": {"image/png": b64, "text/plain": ["<Figure size {}x{}>".format(*[int(v) for v in fig.get_size_inches() * fig.dpi])]},
                    "metadata": {},
                })
                plt.close(fig)

            nb_cells.append({
                "cell_type": "code", "execution_count": exec_count, "metadata": {},
                "outputs": outputs, "source": split_source_lines(text),
            })

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(ipynb_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Built {ipynb_path}  ({len(nb_cells)} cells, {exec_count} executed)")
