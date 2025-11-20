#!/usr/bin/env python3
"""
Утилиты для работы с NVIDIA NeMo диаризацией.
Общий код для transcriber.py и ui.py.
"""

import os
import json
from pathlib import Path

import torch

# Проверяем доступность NeMo
try:
    from nemo.collections.asr.models import ClusteringDiarizer
    from omegaconf import OmegaConf
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False


def get_device():
    """Определяет доступное устройство для вычислений."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_nemo_manifest(audio_path: str, manifest_path: str):
    """
    Создаёт manifest файл для NeMo диаризации.

    Args:
        audio_path: Путь к аудиофайлу
        manifest_path: Путь для сохранения manifest файла
    """
    meta = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None,
        "rttm_filepath": None,
        "uem_filepath": None
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
        f.write('\n')


# Конфигурация NeMo диаризации
# Эти параметры оптимизированы для общего использования
NEMO_CONFIG = {
    # VAD (Voice Activity Detection) параметры
    "vad": {
        "model_path": "vad_multilingual_marblenet",
        "window_length_in_sec": 0.15,
        "shift_length_in_sec": 0.01,
        "smoothing": "median",
        "overlap": 0.5,
        "onset": 0.1,
        "offset": 0.1,
        "pad_onset": 0.1,
        "pad_offset": 0,
        "min_duration_on": 0.2,
        "min_duration_off": 0.2,
        "filter_speech_first": True
    },
    # Speaker Embeddings параметры
    "speaker_embeddings": {
        "model_path": "titanet_large",
        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
        "multiscale_weights": [1, 1, 1, 1, 1],
        "save_embeddings": False
    },
    # Clustering параметры
    "clustering": {
        "oracle_num_speakers": False,
        "max_num_speakers": 8,
        "enhanced_count_thres": 80,
        "max_rp_threshold": 0.25,
        "sparse_search_volume": 30,
        "maj_vote_spk_count": False
    },
    # Общие параметры
    "collar": 0.25,
    "ignore_overlap": True
}


def run_nemo_diarization(audio_path: str, output_dir: str, device: str = None):
    """
    Запускает диаризацию с помощью NeMo.

    Args:
        audio_path: Путь к аудиофайлу
        output_dir: Директория для вывода результатов
        device: Устройство для вычислений ('cuda' или 'cpu')

    Returns:
        list: Список сегментов с информацией о спикерах

    Raises:
        RuntimeError: Если NeMo не установлен
    """
    if not NEMO_AVAILABLE:
        raise RuntimeError("NeMo не установлен. Диаризация недоступна.")

    if device is None:
        device = get_device()

    # Создаём manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    create_nemo_manifest(audio_path, manifest_path)

    # Конфигурация NeMo диаризатора
    config = OmegaConf.create({
        "device": device,
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": output_dir,
            "oracle_vad": False,
            "collar": NEMO_CONFIG["collar"],
            "ignore_overlap": NEMO_CONFIG["ignore_overlap"],

            "vad": {
                "model_path": NEMO_CONFIG["vad"]["model_path"],
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": NEMO_CONFIG["vad"]["window_length_in_sec"],
                    "shift_length_in_sec": NEMO_CONFIG["vad"]["shift_length_in_sec"],
                    "smoothing": NEMO_CONFIG["vad"]["smoothing"],
                    "overlap": NEMO_CONFIG["vad"]["overlap"],
                    "onset": NEMO_CONFIG["vad"]["onset"],
                    "offset": NEMO_CONFIG["vad"]["offset"],
                    "pad_onset": NEMO_CONFIG["vad"]["pad_onset"],
                    "pad_offset": NEMO_CONFIG["vad"]["pad_offset"],
                    "min_duration_on": NEMO_CONFIG["vad"]["min_duration_on"],
                    "min_duration_off": NEMO_CONFIG["vad"]["min_duration_off"],
                    "filter_speech_first": NEMO_CONFIG["vad"]["filter_speech_first"]
                }
            },

            "speaker_embeddings": {
                "model_path": NEMO_CONFIG["speaker_embeddings"]["model_path"],
                "parameters": {
                    "window_length_in_sec": NEMO_CONFIG["speaker_embeddings"]["window_length_in_sec"],
                    "shift_length_in_sec": NEMO_CONFIG["speaker_embeddings"]["shift_length_in_sec"],
                    "multiscale_weights": NEMO_CONFIG["speaker_embeddings"]["multiscale_weights"],
                    "save_embeddings": NEMO_CONFIG["speaker_embeddings"]["save_embeddings"]
                }
            },

            "clustering": {
                "parameters": {
                    "oracle_num_speakers": NEMO_CONFIG["clustering"]["oracle_num_speakers"],
                    "max_num_speakers": NEMO_CONFIG["clustering"]["max_num_speakers"],
                    "enhanced_count_thres": NEMO_CONFIG["clustering"]["enhanced_count_thres"],
                    "max_rp_threshold": NEMO_CONFIG["clustering"]["max_rp_threshold"],
                    "sparse_search_volume": NEMO_CONFIG["clustering"]["sparse_search_volume"],
                    "maj_vote_spk_count": NEMO_CONFIG["clustering"]["maj_vote_spk_count"]
                }
            }
        }
    })

    # Запуск диаризации
    sd_model = ClusteringDiarizer(cfg=config)
    # Устанавливаем атрибут verbose для совместимости с разными версиями NeMo
    if not hasattr(sd_model, 'verbose'):
        sd_model.verbose = False
    sd_model.diarize()

    # Чтение результатов RTTM
    rttm_file = os.path.join(output_dir, "pred_rttms",
                             Path(audio_path).stem + ".rttm")

    diarization_results = []
    if os.path.exists(rttm_file):
        with open(rttm_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    diarization_results.append({
                        "start": start,
                        "end": start + duration,
                        "speaker": speaker
                    })

    return diarization_results


def assign_speakers_to_segments(transcription_segments, diarization_results):
    """
    Назначает спикеров сегментам транскрипции.

    Для каждого сегмента транскрипции находит спикера,
    который говорил в середине этого сегмента.

    Args:
        transcription_segments: Список сегментов транскрипции
        diarization_results: Результаты диаризации

    Returns:
        list: Сегменты транскрипции с назначенными спикерами
    """
    for segment in transcription_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_mid = (seg_start + seg_end) / 2

        # Находим спикера для середины сегмента
        speaker = "SPEAKER_00"
        for diar in diarization_results:
            if diar["start"] <= seg_mid <= diar["end"]:
                speaker = diar["speaker"]
                break

        segment["speaker"] = speaker

    return transcription_segments
