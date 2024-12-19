.PHONY: check
check: ## Run code quality tools.
	@echo "Linting code via pre-commit"
	@pre-commit run -a
