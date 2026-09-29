from pathlib import Path

import nbformat

from notebook_figures import run_notebook


def _write_notebook(path: Path, *sources: str) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(source, metadata={"tags": ["parameters"]} if i == 0 else {})
            for i, source in enumerate(sources)
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, path)


def test_run_notebook_saves_executed_notebook_with_overrides(tmp_path: Path):
    source = tmp_path / "source.ipynb"
    target = tmp_path / "figures"
    _write_notebook(source, "VALUE = 1", "print(f'value={VALUE}')")

    ok, error, _ = run_notebook(source, {"VALUE": 7}, target)

    assert ok
    assert error is None
    output = nbformat.read(target / "output.ipynb", as_version=4)
    assert any("VALUE = 7" in cell.source for cell in output.cells)
    assert any(
        output_item.output_type == "stream" and "value=7" in output_item.text
        for cell in output.cells
        for output_item in cell.get("outputs", [])
    )


def test_run_notebook_saves_partial_notebook_after_cell_failure(tmp_path: Path):
    source = tmp_path / "source.ipynb"
    target = tmp_path / "figures"
    _write_notebook(source, "VALUE = 1", "print('completed')", "raise RuntimeError('expected failure')")

    ok, error, _ = run_notebook(source, {}, target)

    assert not ok
    assert "RuntimeError: expected failure" in error
    output = nbformat.read(target / "output.ipynb", as_version=4)
    assert any(
        output_item.output_type == "stream" and "completed" in output_item.text
        for cell in output.cells
        for output_item in cell.get("outputs", [])
    )
