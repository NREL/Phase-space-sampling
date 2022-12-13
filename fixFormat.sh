poetry update
poetry run isort *py
poetry run black *py
poetry run isort utils
poetry run black utils
poetry run isort data-efficientML
poetry run black data-efficientML
