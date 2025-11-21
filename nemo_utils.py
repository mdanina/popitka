#!/usr/bin/env python3
"""
Утилиты для работы с NVIDIA NeMo диаризацией.
Простой подход на основе извлечения эмбеддингов и кластеризации.
"""

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# Проверяем доступность NeMo
try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False


def get_device():
    """Определяет доступное устройство для вычислений."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Глобальная переменная для кэширования модели
_cached_model = None
_cached_device = None


def _load_speaker_model(device: str):
    """
    Загружает модель для извлечения эмбеддингов спикеров.
    Кэширует модель для повторного использования.
    """
    global _cached_model, _cached_device

    if _cached_model is not None and _cached_device == device:
        return _cached_model

    print("Загрузка модели speaker embeddings (titanet_large)...")

    # Пробуем загрузить модель
    repo = "nvidia/speakerverification_en_titanet_large"
    token = os.getenv("HF_TOKEN")

    try:
        if token:
            model = EncDecSpeakerLabelModel.from_pretrained(repo, token=token)
        else:
            model = EncDecSpeakerLabelModel.from_pretrained(repo)
    except TypeError:
        # Fallback без токена
        model = EncDecSpeakerLabelModel.from_pretrained(repo)

    model = model.to(device).eval()
    _cached_model = model
    _cached_device = device

    return model


def extract_embeddings(wav: np.ndarray, sr: int, model,
                       win_s: float = 3.0, step_s: float = 1.5):
    """
    Извлекает эмбеддинги спикеров из аудио.

    Args:
        wav: Аудио данные (numpy array)
        sr: Частота дискретизации
        model: Модель NeMo для извлечения эмбеддингов
        win_s: Размер окна в секундах
        step_s: Шаг окна в секундах

    Returns:
        tuple: (эмбеддинги, временные метки)
    """
    embs = []
    stamps = []
    t = 0.0
    total_dur = len(wav) / sr

    while t + win_s <= total_dur:
        # Извлекаем сегмент аудио
        segment = wav[int(t * sr): int((t + win_s) * sr)]

        # Создаём временный wav файл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment, sr)
            tmp_path = tmp.name

        try:
            # Получаем эмбеддинг
            with torch.no_grad():
                emb = model.get_embedding(tmp_path).cpu().numpy().squeeze()

            # Нормализуем эмбеддинг
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            embs.append(emb_norm)
            stamps.append((t, t + win_s))
        finally:
            os.remove(tmp_path)

        t += step_s

    if len(embs) == 0:
        return np.array([]), []

    return np.stack(embs), stamps


def auto_cluster(embs: np.ndarray, max_k: int = 8):
    """
    Автоматически определяет число спикеров и кластеризует эмбеддинги.

    Args:
        embs: Матрица эмбеддингов
        max_k: Максимальное число спикеров

    Returns:
        numpy.ndarray: Метки кластеров
    """
    if len(embs) < 2:
        return np.zeros(len(embs), dtype=int)

    # Ограничиваем max_k количеством сэмплов
    max_k = min(max_k, len(embs))

    best_labels = None
    best_score = -1

    # Перебираем число кластеров от 2 до max_k
    for k in range(2, max_k + 1):
        try:
            clustering = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=min(10, len(embs) - 1),
                random_state=42
            )
            labels = clustering.fit_predict(embs)

            # Оцениваем качество кластеризации
            if len(set(labels)) > 1:
                score = silhouette_score(embs, labels)
                if score > best_score:
                    best_labels = labels
                    best_score = score
        except Exception:
            # Пропускаем если кластеризация не удалась
            continue

    # Если ничего не сработало, считаем что 1 спикер
    if best_labels is None:
        return np.zeros(len(embs), dtype=int)

    return best_labels


def merge_segments(stamps, labels, gap: float = 0.5):
    """
    Объединяет последовательные сегменты одного спикера.

    Args:
        stamps: Список временных меток (start, end)
        labels: Метки спикеров
        gap: Максимальный разрыв для объединения

    Returns:
        list: Объединённые сегменты
    """
    if len(stamps) == 0:
        return []

    merged = []
    cur = {"speaker": f"speaker_{int(labels[0])}", "start": stamps[0][0], "end": stamps[0][1]}

    for (s, e), lab in zip(stamps[1:], labels[1:]):
        speaker_label = f"speaker_{int(lab)}"

        # Если тот же спикер и небольшой разрыв - объединяем
        if speaker_label == cur["speaker"] and s <= cur["end"] + gap:
            cur["end"] = e
        else:
            merged.append(cur)
            cur = {"speaker": speaker_label, "start": s, "end": e}

    merged.append(cur)
    return merged


def run_nemo_diarization(audio_path: str, output_dir: str = None, device: str = None,
                         max_speakers: int = 8, window_size: float = 3.0,
                         step_size: float = 1.5):
    """
    Запускает диаризацию с помощью NeMo.

    Args:
        audio_path: Путь к аудиофайлу
        output_dir: Директория для вывода (не используется в новой версии)
        device: Устройство для вычислений ('cuda' или 'cpu')
        max_speakers: Максимальное число спикеров
        window_size: Размер окна для извлечения эмбеддингов (секунды)
        step_size: Шаг окна (секунды)

    Returns:
        list: Список сегментов с информацией о спикерах

    Raises:
        RuntimeError: Если NeMo не установлен
    """
    if not NEMO_AVAILABLE:
        raise RuntimeError("NeMo не установлен. Диаризация недоступна.")

    if device is None:
        device = get_device()

    # Загружаем модель
    model = _load_speaker_model(device)

    # Читаем аудио
    print("Чтение аудио...")
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Проверяем длительность
    duration = len(wav) / sr
    if duration < window_size:
        print(f"Аудио слишком короткое ({duration:.1f}s < {window_size}s). Один спикер.")
        return [{"start": 0.0, "end": duration, "speaker": "speaker_0"}]

    # Извлекаем эмбеддинги
    print("Извлечение эмбеддингов...")
    embs, stamps = extract_embeddings(wav, sr, model, window_size, step_size)

    if len(embs) == 0:
        print("Не удалось извлечь эмбеддинги. Один спикер.")
        return [{"start": 0.0, "end": duration, "speaker": "speaker_0"}]

    # Кластеризуем
    print(f"Кластеризация (до {max_speakers} спикеров)...")
    labels = auto_cluster(embs, max_k=max_speakers)

    num_speakers = len(set(labels))
    print(f"Найдено спикеров: {num_speakers}")

    # Объединяем сегменты
    diarization_results = merge_segments(stamps, labels)

    return diarization_results


def assign_speakers_to_segments(transcription_segments, diarization_results):
    """
    Назначает спикеров сегментам транскрипции.

    Использует пересечение интервалов для более точного определения спикера.

    Args:
        transcription_segments: Список сегментов транскрипции
        diarization_results: Результаты диаризации

    Returns:
        list: Сегменты транскрипции с назначенными спикерами
    """
    if not diarization_results:
        # Если нет результатов диаризации, назначаем всем один спикер
        for segment in transcription_segments:
            segment["speaker"] = "speaker_0"
        return transcription_segments

    for segment in transcription_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        # Находим спикера по пересечению интервалов
        speaker = "speaker_0"
        max_overlap = 0

        for diar in diarization_results:
            # Вычисляем пересечение интервалов
            overlap_start = max(seg_start, diar["start"])
            overlap_end = min(seg_end, diar["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                speaker = diar["speaker"]

        # Fallback: если нет пересечения, ищем по середине сегмента
        if max_overlap == 0:
            seg_mid = (seg_start + seg_end) / 2
            for diar in diarization_results:
                if diar["start"] <= seg_mid <= diar["end"]:
                    speaker = diar["speaker"]
                    break

        segment["speaker"] = speaker

    return transcription_segments


def preload_models(device: str = None):
    """
    Предзагружает модели NeMo в память.
    Вызывается для ускорения первой диаризации.

    Args:
        device: Устройство для загрузки модели
    """
    if not NEMO_AVAILABLE:
        return

    if device is None:
        device = get_device()

    _load_speaker_model(device)
    print("Модель speaker embeddings загружена.")
