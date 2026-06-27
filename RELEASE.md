# Release checklist

Before release:

1. Update version in pyproject.toml.
2. Update CHANGELOG.md.
3. Run:
   python -m pip install -e ".[dev,torch]"
   ruff check .
   ruff format --check .
   mypy src/hyperglyph
   pytest --cov=hyperglyph
   python -m build
   twine check dist/*
4. Test CLI locally:
   hyperglyph --help
5. Push to GitHub.
6. Confirm CI passes.
7. Create GitHub release:
   tag: v0.1.0
8. GitHub Actions publish.yml publishes to PyPI.
9. Verify:
   pip install hyperglyph
   python -c "import hyperglyph; print(hyperglyph.__version__)"
