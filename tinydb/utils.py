[tool.poetry]
name = "tinydb"
version = "0.1.0"
description = ""
authors = ["Somasree Majumder <bishnupadamajumder32@gmail.com@gmail.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.12"
numpy = "^1.26.2"
networkx = "^3.2.1"
tqdm = "^4.66.1"
transformers = {version = "^4.35.2", optional = true}
sentence-transformers = {version = "^2.2.2", optional = true}
pillow = {version = "^10.1.0", optional = true}
requests = {version = "^2.31.0", optional = true}
datasets = {version = "^2.15.0", optional = true}

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
matplotlib = "^3.8.2"

[tool.poetry.extras]
examples = ["transformers", "sentence-transformers", "pillow", "requests", "datasets"]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"