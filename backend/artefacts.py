import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import mode
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class CTGAnalyzer:
    """
    Анализатор сигнала КТГ для выявления сущностей и артефактов
    """
    
    def __init__(self, sampling_rate: float = 4.0):
        """
        Инициализация анализатора
        
        Args:
            sampling_rate: Частота дискретизации сигнала (Гц)
        """
        self.sampling_rate = sampling_rate
        self.epoch_length = int(10 * sampling_rate)  # 10-минутные эпохи
        
        # Пороговые значения
        self.bradycardia_threshold = 110
        self.tachycardia_threshold = 160
        self.accel_threshold = 15  # уд/мин
        self.accel_duration_min = 15  # секунд
        self.accel_duration_max = 600  # секунд
        self.decel_threshold = 15  # уд/мин
        self.decel_duration_min = 15  # секунд
        self.variability_normal = (5, 25)  # нормальная вариабельность
        
    def preprocess_signal(self, fhr_signal: np.ndarray, ua_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предобработка сигналов КТГ
        
        Args:
            fhr_signal: Сигнал ЧСС плода
            ua_signal: Сигнал маточной активности
            
        Returns:
            Очищенные сигналы ЧСС и маточной активности
        """
        # Копируем сигналы чтобы не изменять оригиналы
        fhr_clean = fhr_signal.copy().astype(float)
        ua_clean = ua_signal.copy().astype(float)
        
        # 1. Обнаружение и маркировка артефактов
        artifact_mask = self._detect_artifacts(fhr_clean)
        
        # 2. Фильтрация сигналов
        # Баттерворт фильтр 0.03-3 Гц для ЧСС
        nyquist = 0.5 * self.sampling_rate
        low = 0.03 / nyquist
        high = 3.0 / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        fhr_clean[~artifact_mask] = signal.filtfilt(b, a, fhr_clean[~artifact_mask])
        
        # 3. Интерполяция артефактов
        if np.any(artifact_mask):
            indices = np.arange(len(fhr_clean))
            valid_indices = indices[~artifact_mask]
            valid_values = fhr_clean[~artifact_mask]
            if len(valid_values) > 1:
                fhr_clean[artifact_mask] = np.interp(indices[artifact_mask], valid_indices, valid_values)
        
        return fhr_clean, ua_clean, artifact_mask
    
    def _detect_artifacts(self, fhr_signal: np.ndarray) -> np.ndarray:
        """
        Обнаружение артефактов в сигнале ЧСС
        
        Returns:
            Маска артефактов (True где артефакт)
        """
        artifact_mask = np.zeros_like(fhr_signal, dtype=bool)
        
        # 1. Потеря сигнала (нулевые или постоянные значения)
        signal_loss = (fhr_signal == 0) | (fhr_signal < 50) | (fhr_signal > 240)
        artifact_mask |= signal_loss
        
        # 2. Резкие скачки (большие производные)
        diff_signal = np.abs(np.diff(fhr_signal, prepend=fhr_signal[0]))
        spikes = diff_signal > 25  # порог для скачков
        artifact_mask |= spikes
        
        # 3. Отсутствие вариабельности (плато)
        window_size = int(30 * self.sampling_rate)  # 30-секундные окна
        if len(fhr_signal) > window_size:
            for i in range(0, len(fhr_signal) - window_size, window_size//2):
                window = fhr_signal[i:i + window_size]
                if np.std(window) < 0.5:  # очень низкая вариабельность
                    artifact_mask[i:i + window_size] = True
        
        return artifact_mask
    
    def calculate_baseline(self, fhr_signal: np.ndarray, window_minutes: int = 10) -> np.ndarray:
        """
        Расчет базальной ЧСС скользящим окном
        
        Args:
            fhr_signal: Сигнал ЧСС
            window_minutes: Размер окна в минутах
            
        Returns:
            Массив базальной ЧСС
        """
        window_size = int(window_minutes * 60 * self.sampling_rate)
        baseline = np.zeros_like(fhr_signal)
        
        for i in range(len(fhr_signal)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(fhr_signal), i + window_size // 2)
            
            window = fhr_signal[start_idx:end_idx]
            if len(window) > 0:
                # Используем медиану для устойчивости к выбросам
                baseline[i] = np.median(window)
        
        return baseline
    
    def detect_accelerations(self, fhr_signal: np.ndarray, baseline: np.ndarray) -> List[Dict]:
        """
        Детекция акцелераций
        
        Returns:
            Список словарей с информацией об акцелерациях
        """
        accelerations = []
        
        # Вычисляем отклонение от базального ритма
        deviation = fhr_signal - baseline
        
        # Находим участки выше порога
        above_threshold = deviation > self.accel_threshold
        
        if not np.any(above_threshold):
            return accelerations
        
        # Находим границы акцелераций
        diff_above = np.diff(above_threshold.astype(int))
        start_indices = np.where(diff_above == 1)[0] + 1
        end_indices = np.where(diff_above == -1)[0] + 1
        
        # Обрабатываем каждую акцелерацию
        for start_idx, end_idx in zip(start_indices, end_indices):
            duration_sec = (end_idx - start_idx) / self.sampling_rate
            
            if (duration_sec >= self.accel_duration_min and 
                duration_sec <= self.accel_duration_max):
                
                accel_amplitude = np.max(fhr_signal[start_idx:end_idx]) - baseline[start_idx]
                
                acceleration = {
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'start_time': start_idx / self.sampling_rate,
                    'end_time': end_idx / self.sampling_rate,
                    'duration_sec': duration_sec,
                    'amplitude': accel_amplitude,
                    'max_fhr': np.max(fhr_signal[start_idx:end_idx])
                }
                accelerations.append(acceleration)
        
        return accelerations
    
    def detect_decelerations(self, fhr_signal: np.ndarray, baseline: np.ndarray, 
                           ua_signal: np.ndarray = None) -> List[Dict]:
        """
        Детекция и классификация децелераций
        
        Returns:
            Список словарей с информацией о децелерациях
        """
        decelerations = []
        
        # Вычисляем отклонение от базального ритма
        deviation = baseline - fhr_signal  # отрицательное отклонение
        
        # Находим участки ниже порога
        below_threshold = deviation > self.decel_threshold
        
        if not np.any(below_threshold):
            return decelerations
        
        # Находим границы децелераций
        diff_below = np.diff(below_threshold.astype(int))
        start_indices = np.where(diff_below == 1)[0] + 1
        end_indices = np.where(diff_below == -1)[0] + 1
        
        for start_idx, end_idx in zip(start_indices, end_indices):
            duration_sec = (end_idx - start_idx) / self.sampling_rate
            
            if duration_sec >= self.decel_duration_min:
                
                decel_amplitude = baseline[start_idx] - np.min(fhr_signal[start_idx:end_idx])
                decel_type = self._classify_deceleration(fhr_signal, baseline, ua_signal, 
                                                       start_idx, end_idx)
                
                deceleration = {
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'start_time': start_idx / self.sampling_rate,
                    'end_time': end_idx / self.sampling_rate,
                    'duration_sec': duration_sec,
                    'amplitude': decel_amplitude,
                    'min_fhr': np.min(fhr_signal[start_idx:end_idx]),
                    'type': decel_type,
                    'severity': self._assess_deceleration_severity(decel_amplitude, duration_sec)
                }
                decelerations.append(deceleration)
        
        return decelerations
    
    def _classify_deceleration(self, fhr_signal: np.ndarray, baseline: np.ndarray,
                             ua_signal: np.ndarray, start_idx: int, end_idx: int) -> str:
        """
        Классификация типа децелерации
        """
        if ua_signal is None:
            return "variable"  # без сигнала UA сложно классифицировать
        
        # Анализ формы децелерации
        decel_segment = fhr_signal[start_idx:end_idx]
        time_to_nadir = np.argmin(decel_segment) / self.sampling_rate  # время до дна
        
        # Анализ связи со схватками
        ua_window = ua_signal[max(0, start_idx-300):min(len(ua_signal), end_idx+300)]
        contraction_times = self._find_contractions(ua_window)
        
        # Простая классификация на основе времени и формы
        if time_to_nadir < 30:  # быстрый спуск
            return "variable"
        else:
            return "late"  # упрощенная классификация
    
    def _find_contractions(self, ua_signal: np.ndarray) -> List[int]:
        """
        Поиск схваток в сигнале маточной активности
        """
        contractions = []
        threshold = np.percentile(ua_signal, 80)  # порог для схваток
        
        above_threshold = ua_signal > threshold
        diff_above = np.diff(above_threshold.astype(int))
        start_indices = np.where(diff_above == 1)[0]
        
        return start_indices.tolist()
    
    def _assess_deceleration_severity(self, amplitude: float, duration: float) -> str:
        """
        Оценка тяжести децелерации
        """
        if amplitude <= 15:
            return "mild"
        elif amplitude <= 45:
            return "moderate"
        else:
            return "severe"
    
    def calculate_variability(self, fhr_signal: np.ndarray, window_minutes: int = 1) -> Dict[str, np.ndarray]:
        """
        Расчет вариабельности сердечного ритма
        
        Returns:
            Словарь с показателями вариабельности
        """
        window_size = int(window_minutes * 60 * self.sampling_rate)
        n_windows = len(fhr_signal) // window_size
        
        stv_values = []  # кратковременная вариабельность
        ltv_amplitude_values = []  # амплитуда долговременной вариабельности
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window = fhr_signal[start_idx:end_idx]
            
            if len(window) < 2:
                continue
            
            # Кратковременная вариабельность (STV)
            differences = np.abs(np.diff(window))
            stv = np.mean(differences) if len(differences) > 0 else 0
            stv_values.extend([stv] * len(window[start_idx:end_idx]))
            
            # Амплитуда долговременной вариабельности
            ltv_amp = np.max(window) - np.min(window) if len(window) > 0 else 0
            ltv_amplitude_values.extend([ltv_amp] * len(window[start_idx:end_idx]))
        
        # Дополняем до исходной длины
        if len(stv_values) < len(fhr_signal):
            stv_values.extend([stv_values[-1] if stv_values else 0] * 
                            (len(fhr_signal) - len(stv_values)))
        if len(ltv_amplitude_values) < len(fhr_signal):
            ltv_amplitude_values.extend([ltv_amplitude_values[-1] if ltv_amplitude_values else 0] * 
                                      (len(fhr_signal) - len(ltv_amplitude_values)))
        
        return {
            'stv': np.array(stv_values[:len(fhr_signal)]),
            'ltv_amplitude': np.array(ltv_amplitude_values[:len(fhr_signal)])
        }
    
    def assess_fisher_score(self, analysis_results: Dict) -> Dict:
        """
        Оценка по шкале Фишера
        """
        score = 0
        details = {}
        
        # Базальный ритм
        avg_baseline = np.mean(analysis_results['baseline'])
        if 120 <= avg_baseline <= 160:
            score += 2
            details['baseline'] = 2
        elif 100 <= avg_baseline <= 120 or 160 <= avg_baseline <= 180:
            score += 1
            details['baseline'] = 1
        else:
            details['baseline'] = 0
        
        # Вариабельность (амплитуда)
        avg_variability = np.mean(analysis_results['variability']['ltv_amplitude'])
        if 6 <= avg_variability <= 25:
            score += 2
            details['variability_amp'] = 2
        elif 3 <= avg_variability <= 5:
            score += 1
            details['variability_amp'] = 1
        else:
            details['variability_amp'] = 0
        
        # Акцелерации
        n_accelerations = len(analysis_results['accelerations'])
        if n_accelerations >= 2:
            score += 2
            details['accelerations'] = 2
        elif n_accelerations == 1:
            score += 1
            details['accelerations'] = 1
        else:
            details['accelerations'] = 0
        
        # Децелерации
        n_decelerations = len([d for d in analysis_results['decelerations'] 
                             if d['severity'] in ['moderate', 'severe']])
        if n_decelerations == 0:
            score += 2
            details['decelerations'] = 2
        elif n_decelerations <= 2:
            score += 1
            details['decelerations'] = 1
        else:
            details['decelerations'] = 0
        
        # Интерпретация
        if score >= 8:
            interpretation = "Нормальное состояние плода"
        elif score >= 5:
            interpretation = "Начальные признаки гипоксии"
        else:
            interpretation = "Выраженная гипоксия"
        
        return {
            'total_score': score,
            'details': details,
            'interpretation': interpretation
        }
    
    def analyze_complete_ctg(self, fhr_signal: np.ndarray, ua_signal: np.ndarray = None, 
                           duration_minutes: int = 20) -> Dict:
        """
        Полный анализ КТГ записи
        
        Args:
            fhr_signal: Сигнал ЧСС плода
            ua_signal: Сигнал маточной активности
            duration_minutes: Длина анализа в минутах
            
        Returns:
            Словарь с результатами анализа
        """
        # Ограничиваем длину сигнала для анализа
        max_samples = int(duration_minutes * 60 * self.sampling_rate)
        if len(fhr_signal) > max_samples:
            fhr_signal = fhr_signal[:max_samples]
            if ua_signal is not None:
                ua_signal = ua_signal[:max_samples]
        
        # Предобработка
        fhr_clean, ua_clean, artifact_mask = self.preprocess_signal(fhr_signal, ua_signal)
        
        # Расчет базовых параметров
        baseline = self.calculate_baseline(fhr_clean)
        variability = self.calculate_variability(fhr_clean)
        
        # Детекция событий
        accelerations = self.detect_accelerations(fhr_clean, baseline)
        decelerations = self.detect_decelerations(fhr_clean, baseline, ua_clean)
        
        # Оценка по Фишеру
        fisher_score = self.assess_fisher_score({
            'baseline': baseline,
            'variability': variability,
            'accelerations': accelerations,
            'decelerations': decelerations
        })
        
        # Статистика
        avg_fhr = np.mean(fhr_clean)
        std_fhr = np.std(fhr_clean)
        
        return {
            'signals': {
                'fhr_clean': fhr_clean,
                'ua_clean': ua_clean,
                'baseline': baseline,
                'artifact_mask': artifact_mask
            },
            'variability': variability,
            'events': {
                'accelerations': accelerations,
                'decelerations': decelerations
            },
            'statistics': {
                'mean_fhr': avg_fhr,
                'std_fhr': std_fhr,
                'n_accelerations': len(accelerations),
                'n_decelerations': len(decelerations),
                'artifact_percentage': np.mean(artifact_mask) * 100
            },
            'clinical_scores': {
                'fisher': fisher_score
            }
        }

# Пример использования
def demo_ctg_analysis():
    """Демонстрация работы анализатора КТГ"""
    
    # Генерация тестового сигнала
    sampling_rate = 4.0  # 4 Hz
    duration = 1200  # 20 минут в секундах
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Базовый сигнал ЧСС с вариабельностью
    baseline_fhr = 140 + 10 * np.sin(2 * np.pi * t / 300)  # медленные колебания
    variability = 8 * np.random.randn(len(t))  # случайная вариабельность
    fhr_signal = baseline_fhr + variability
    
    # Добавляем акцелерации
    accel_times = [300, 600, 900]  # секунды
    for time in accel_times:
        idx = int(time * sampling_rate)
        duration_accel = int(30 * sampling_rate)  # 30 секунд
        fhr_signal[idx:idx + duration_accel] += 20
    
    # Добавляем децелерации
    decel_times = [450, 750]
    for time in decel_times:
        idx = int(time * sampling_rate)
        duration_decel = int(45 * sampling_rate)  # 45 секунд
        fhr_signal[idx:idx + duration_decel] -= 25
    
    # Добавляем артефакты
    artifact_times = [200, 800]
    for time in artifact_times:
        idx = int(time * sampling_rate)
        duration_artifact = int(10 * sampling_rate)  # 10 секунд
        fhr_signal[idx:idx + duration_artifact] = 0
    
    # Сигнал маточной активности (схватки)
    ua_signal = np.zeros_like(t)
    contraction_times = [100, 400, 700, 1000]
    for time in contraction_times:
        idx = int(time * sampling_rate)
        duration_contraction = int(60 * sampling_rate)  # 60 секунд
        ua_signal[idx:idx + duration_contraction] = 50 * np.sin(
            np.pi * np.linspace(0, 1, duration_contraction)
        )
    
    # Анализ КТГ
    analyzer = CTGAnalyzer(sampling_rate=sampling_rate)
    results = analyzer.analyze_complete_ctg(fhr_signal, ua_signal)
    
    # Вывод результатов
    print("=== РЕЗУЛЬТАТЫ АНАЛИЗА КТГ ===")
    print(f"Средняя ЧСС: {results['statistics']['mean_fhr']:.1f} уд/мин")
    print(f"Количество акцелераций: {results['statistics']['n_accelerations']}")
    print(f"Количество децелераций: {results['statistics']['n_decelerations']}")
    print(f"Артефакты: {results['statistics']['artifact_percentage']:.1f}%")
    print(f"Оценка по Фишеру: {results['clinical_scores']['fisher']['total_score']} баллов")
    print(f"Интерпретация: {results['clinical_scores']['fisher']['interpretation']}")
    
    # Детали по децелерациям
    if results['events']['decelerations']:
        print("\n--- Децелерации ---")
        for i, decel in enumerate(results['events']['decelerations']):
            print(f"Децелерация {i+1}: время {decel['start_time']:.0f}-{decel['end_time']:.0f}с, "
                  f"амплитуда {decel['amplitude']:.1f} уд/мин, тип: {decel['type']}, "
                  f"тяжесть: {decel['severity']}")
    
    return results

if __name__ == "__main__":
    results = demo_ctg_analysis()