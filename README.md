# Локальный Транскрибатор (Whisper + NeMo)

Офлайн транскрибатор аудио с диаризацией спикеров. Использует Faster-Whisper для транскрибации и NVIDIA NeMo для диаризации.

**Не требует регистрации или токенов!**

## Возможности

- Транскрибация длинных аудиофайлов (1+ час)
- Диаризация спикеров (NVIDIA NeMo)
- Поддержка русского и английского языка
- Работа полностью офлайн (после скачивания моделей)
- Оптимизация для GPU (RTX 4050 и выше)
- Форматы вывода: TXT, SRT, JSON

## Требования

- Python 3.8-3.10
- NVIDIA GPU с 6+ GB VRAM (рекомендуется)
- CUDA 11.8 или 12.x
- FFmpeg
- 8+ GB RAM

## Установка

### 1. Установка CUDA и cuDNN

Убедитесь что установлены CUDA drivers. Проверка:
```bash
nvidia-smi
```

### 2. Установка FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Скачайте с https://ffmpeg.org/download.html и добавьте в PATH.

### 3. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux
# или
venv\Scripts\activate  # Windows
```

### 4. Установка PyTorch с CUDA

```bash
# Для CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Для CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 6. Первый запуск (скачивание моделей)

При первом запуске модели скачаются автоматически (~3-5 GB):
```bash
python transcriber.py test_audio.mp3 -l ru
```

После этого всё работает офлайн.

## Использование

### Базовое использование

```bash
# Транскрибация на русском с диаризацией
python transcriber.py audio.mp3 -l ru

# Транскрибация на английском
python transcriber.py audio.mp3 -l en

# Без диаризации (быстрее)
python transcriber.py audio.mp3 -l ru --no-diarize
```

### Настройки для RTX 4050 (6 GB VRAM)

```bash
# Оптимальные настройки
python transcriber.py audio.mp3 -l ru -m small -c float16

# Максимальное качество (может не хватить памяти)
python transcriber.py audio.mp3 -l ru -m medium -c float16
```

### Форматы вывода

```bash
# Субтитры SRT
python transcriber.py audio.mp3 -l ru -f srt

# JSON (машиночитаемый)
python transcriber.py audio.mp3 -l ru -f json

# Текст (по умолчанию)
python transcriber.py audio.mp3 -l ru -f txt
```

### Все опции

```bash
python transcriber.py --help
```

| Опция | Описание | По умолчанию |
|-------|----------|--------------|
| `-l, --language` | Язык (ru/en) | ru |
| `-m, --model` | Размер модели Whisper | small |
| `--diarize/--no-diarize` | Диаризация | включена |
| `-o, --output` | Путь вывода | автогенерация |
| `-f, --format` | Формат (txt/srt/json) | txt |
| `-b, --beam-size` | Beam size | 5 |
| `-c, --compute-type` | Тип вычислений | float16 |

## Выбор модели Whisper

| Модель | VRAM | Качество | Скорость |
|--------|------|----------|----------|
| tiny | ~1 GB | Низкое | Очень быстро |
| base | ~1 GB | Среднее | Быстро |
| **small** | ~2 GB | Хорошее | Средне |
| medium | ~5 GB | Отличное | Медленно |
| large-v3 | ~10 GB | Лучшее | Медленно |

**Для RTX 4050 (6 GB):** используйте `small` или `medium`.

## Примерное время обработки

Для 1 часа аудио на RTX 4050:
- Whisper small: ~3-5 минут
- NeMo диаризация: ~5-10 минут
- **Итого: ~10-15 минут**

На CPU будет в 10-20 раз медленнее.

## Примеры вывода

### TXT формат
```
[00:00:01.200 - 00:00:05.800] speaker_0: Добрый день, начинаем совещание.
[00:00:06.100 - 00:00:12.400] speaker_1: Да, давайте обсудим первый вопрос.
```

### SRT формат
```
1
00:00:01,200 --> 00:00:05,800
[speaker_0] Добрый день, начинаем совещание.

2
00:00:06,100 --> 00:00:12,400
[speaker_1] Да, давайте обсудим первый вопрос.
```

## Решение проблем

### CUDA out of memory
- Используйте меньшую модель: `-m small` вместо `-m medium`
- Уменьшите beam_size: `-b 3`
- Закройте другие приложения использующие GPU

### Медленная работа
- Убедитесь что используется GPU: должно показывать "CUDA"
- Проверьте `nvidia-smi` во время работы

### Ошибка импорта NeMo
```bash
pip install nemo_toolkit[asr] --upgrade
```

### Ошибка с librosa/soundfile
```bash
pip install soundfile librosa --upgrade
```

## Архитектура

1. **Faster-Whisper** - быстрая транскрибация на GPU с CTranslate2
2. **NeMo TitaNet** - speaker embeddings для диаризации
3. **NeMo VAD** - определение речевой активности
4. **Multiscale Clustering** - кластеризация спикеров

## Преимущества перед WhisperX

- **Не нужен HuggingFace токен**
- **Не нужно принимать лицензии**
- Полностью open source
- Лучшее качество диаризации в некоторых случаях

## Ограничения

- Требуется NVIDIA GPU для комфортной работы
- Первый запуск требует интернет для скачивания моделей
- NeMo - большая библиотека (~5 GB)

## Лицензия

MIT
