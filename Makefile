.PHONY: install run clean

# Установка зависимостей через uv
install-dev:
	uv sync

install:
	uv sync --no-dev
# Запуск API в режиме разработки (с автоперезагрузкой)
run:
	uv run uvicorn main:app --reload --port 8000

# Очистка временных файлов
clean:
	rm -rf __pycache__ .pytest_cache .uv