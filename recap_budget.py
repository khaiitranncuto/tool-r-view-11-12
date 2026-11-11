"""Initial duration calibration; sentence counts remain an exact contract."""
import math

DEFAULT_VOICE_RATE = '+20%'
CALIBRATED_SENTENCES_PER_MINUTE = 13.5
CALIBRATED_RATE_FACTOR = 1.2
WORDS_PER_SENTENCE = 20
BUDGET_VERSION = 3


def normalize_voice_rate(value=DEFAULT_VOICE_RATE):
    try:
        rate = int(str(value).strip().removesuffix('%'))
    except (ValueError, TypeError):
        raise ValueError('Tốc độ đọc phải là số phần trăm nguyên, ví dụ +20%.')
    if not -50 <= rate <= 100:
        raise ValueError('Tốc độ đọc phải nằm trong khoảng -50% đến +100%.')
    return f'{rate:+d}%'


def sentences_per_minute(voice_rate=DEFAULT_VOICE_RATE):
    rate = int(normalize_voice_rate(voice_rate)[:-1])
    return CALIBRATED_SENTENCES_PER_MINUTE * (1 + rate / 100) / CALIBRATED_RATE_FACTOR


def narration_budget(minutes, voice_rate=DEFAULT_VOICE_RATE):
    minutes = float(minutes)
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError('Số phút recap phải là số dương hữu hạn.')
    rate = normalize_voice_rate(voice_rate)
    coefficient = sentences_per_minute(rate)
    raw_words = minutes * coefficient * WORDS_PER_SENTENCE
    count = max(3, math.floor(raw_words / WORDS_PER_SENTENCE + .5))
    return {'version': BUDGET_VERSION, 'target_minutes': minutes, 'voice_rate': rate,
            'sentences_per_minute': coefficient, 'target_sentence_count': count,
            'target_words': count * WORDS_PER_SENTENCE, 'estimated': True}


def measured_narration_timing(count, durations, voice_rate, budget=None):
    seconds = sum(max(0.0, float(value)) for value in durations)
    rate = normalize_voice_rate(voice_rate)
    minutes = seconds / 60
    measured = count / minutes if minutes > 0 else 0.0
    return {'actual_sentences': count, 'actual_seconds': round(seconds, 3),
            'actual_minutes': round(minutes, 3), 'voice_rate': rate,
            'target_minutes': (budget or {}).get('target_minutes'),
            'target_sentence_count': (budget or {}).get('target_sentence_count'),
            'measured_sentences_per_minute': round(measured, 4),
            'equivalent_sentences_per_minute_at_20': round(
                measured * CALIBRATED_RATE_FACTOR / (1 + int(rate[:-1]) / 100), 4)}
