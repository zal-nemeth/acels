action-pytest:
    # command to run your tests
	pytest tests

action-black:
    # command to run black for code formatting checks
	black --check .

action-isort:
    # command to check the import sorting
	isort --check-only .