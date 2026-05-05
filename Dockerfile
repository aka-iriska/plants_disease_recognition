FROM python:3.12-slim

WORKDIR /app

# Устанавливаем системные зависимости, необходимые для OpenCV и работы с видео
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Копируем конфигурационные файлы проекта
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости с помощью uv (только основные, без torch/torchvision из dev)
RUN uv sync --no-dev --frozen

# Копируем исходный код
COPY ./src /app/src
COPY ./weights /app/weights

# Запуск через виртуальное окружение uv
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
