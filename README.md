Poetry setup for Pietro

This repository uses Poetry for Python dependency and environment management.

Quick start

1. Install Poetry (follow https://python-poetry.org/docs/#installation).

2. From the project root, create and enter the virtual environment:

```
poetry install
poetry shell
```

3. Run the application inside the poetry shell:

```
python main.py
```

Notes

- `requirements.txt` is preserved for users who don't use Poetry.
- The `pyproject.toml` pins package names but not versions (you can change them as needed).
 - The `requirements.txt` file is preserved for users who prefer pip. To export Poetry lock to requirements for deployment, run:

```
poetry export -f requirements.txt --output requirements.txt --without-hashes
```
