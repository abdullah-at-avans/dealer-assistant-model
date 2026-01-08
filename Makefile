PYTHON=python3
REQUIREMENTS=requirements.txt
VENV=.venv
PIP=$(VENV)/bin/pip

create-venv:
	@$(PYTHON) -m venv $(VENV)

install:
	@$(PIP) install --no-cache-dir -r ./$(REQUIREMENTS)

test:
	@$(PYTHON) test_assistant.py

export-model:
	@$(PYTHON) export_model.py