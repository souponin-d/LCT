"""
Модуль для комплексного анализа сигналов КТГ (кардиотокографии)
Выявление всех сущностей и классификация состояний плода
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CTGAnalyzer:
    """
    Комплексный анализатор сигналов КТГ для выявления сущностей и классификации состояний
    """
    
    def __init__(self, sampling_rate=4.0):
        """
        Инициализация анализатора
        
        Args:
            sampling_rate (float): Частота дискретизации в Гц (обычно 4 Гц для КТГ)
        """
        self.sampling_rate = sampling_rate
        self.window_size_10min = int(10 * 60 * sampling_rate)  # 10 минут в отсчетах
        self.window_size_1min = int(1 * 60 * sampling_rate)    # 1 минута в отсчетах
        
    def preprocess_signal(self, fhr_signal, uc_signal=None):
        """
        Предобработка сигналов КТГ
        
        Args:
            fhr_signal (array): Сигнал ЧСС плода (FHR - Fetal Heart Rate)
            uc_signal (array): Сигнал маточных сокращений (UC - Uterine Contractions)
            
        Returns:
            dict: Предобработанные сигналы и метаданные
        """
        # Удаление NaN и заполнение пропусков
        fhr_clean = pd.Series(fhr_signal).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Сглаживание для удаления высокочастотного шума
        fhr_smoothed = gaussian_filter1d(fhr_clean, sigma=1.0)
        
        # Обработка UC сигнала если есть
        uc_clean = None
        if uc_signal is not None:
            uc_clean = pd.Series(uc_signal).interpolate(method='linear').fillna(0)
            uc_clean = gaussian_filter1d(uc_clean, sigma=1.0)
        
        return {
            'fhr_raw': np.array(fhr_signal),
            'fhr_clean': np.array(fhr_clean),
            'fhr_smoothed': fhr_smoothed,
            'uc_clean': uc_clean,
            'time_axis': np.arange(len(fhr_signal)) / self.sampling_rate
        }
    
    def detect_artifacts(self, fhr_signal):
        """
        Детекция артефактов в сигнале КТГ
        
        Args:
            fhr_signal (array): Сигнал ЧСС
            
        Returns:
            dict: Информация об артефактах
        """
        artifacts = {
            'signal_loss': [],
            'extreme_values': [],
            'high_frequency_noise': [],
            'total_artifact_percentage': 0
        }
        
        # 1. Потеря сигнала (постоянные значения или нули)
        for i in range(len(fhr_signal) - self.window_size_1min):
            window = fhr_signal[i:i + self.window_size_1min]
            if np.std(window) < 1.0 or np.mean(window) == 0:
                artifacts['signal_loss'].append((i, i + self.window_size_1min))
        
        # 2. Экстремальные выбросы (нефизиологические значения)
        extreme_indices = np.where((fhr_signal < 50) | (fhr_signal > 220))[0]
        if len(extreme_indices) > 0:
            # Группировка соседних индексов
            groups = []
            current_group = [extreme_indices[0]]
            for i in range(1, len(extreme_indices)):
                if extreme_indices[i] - extreme_indices[i-1] <= self.sampling_rate:  # 1 секунда
                    current_group.append(extreme_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [extreme_indices[i]]
            groups.append(current_group)
            
            artifacts['extreme_values'] = [(group[0], group[-1]) for group in groups]
        
        # 3. Высокочастотный шум
        # Анализ спектра для детекции шума
        freqs, psd = signal.welch(fhr_signal, fs=self.sampling_rate, nperseg=min(len(fhr_signal)//4, 256))
        high_freq_power = np.sum(psd[freqs > 1.0])  # Мощность выше 1 Гц
        total_power = np.sum(psd)
        
        if high_freq_power / total_power > 0.3:  # Если >30% мощности в высоких частотах
            artifacts['high_frequency_noise'] = [(0, len(fhr_signal))]
        
        # Подсчет общего процента артефактов
        total_artifact_samples = 0
        for start, end in artifacts['signal_loss'] + artifacts['extreme_values']:
            total_artifact_samples += (end - start)
        
        artifacts['total_artifact_percentage'] = (total_artifact_samples / len(fhr_signal)) * 100
        
        return artifacts
    
    def calculate_baseline_fhr(self, fhr_signal, exclude_accelerations=True, exclude_decelerations=True):
        """
        Расчет базальной частоты сердечных сокращений (БЧСС)
        
        Args:
            fhr_signal (array): Сигнал ЧСС
            exclude_accelerations (bool): Исключать акцелерации
            exclude_decelerations (bool): Исключать децелерации
            
        Returns:
            dict: Информация о БЧСС
        """
        baseline_values = []
        baseline_segments = []
        
        # Анализ по 10-минутным окнам
        for i in range(0, len(fhr_signal) - self.window_size_10min, self.window_size_10min):
            window = fhr_signal[i:i + self.window_size_10min]
            
            # Предварительная оценка базального уровня
            preliminary_baseline = np.median(window)
            
            # Исключение акцелераций и децелераций
            if exclude_accelerations or exclude_decelerations:
                mask = np.ones(len(window), dtype=bool)
                
                if exclude_accelerations:
                    # Исключаем участки выше baseline + 15 уд/мин
                    mask &= (window <= preliminary_baseline + 15)
                
                if exclude_decelerations:
                    # Исключаем участки ниже baseline - 15 уд/мин
                    mask &= (window >= preliminary_baseline - 15)
                
                # Оставляем только "спокойные" участки
                if np.sum(mask) > len(window) * 0.3:  # Минимум 30% окна должно остаться
                    window_filtered = window[mask]
                else:
                    window_filtered = window  # Если слишком много исключений, используем все
            else:
                window_filtered = window
            
            # Расчет базального значения как моды или медианы
            try:
                # Попытка найти моду (наиболее частое значение)
                hist, bin_edges = np.histogram(window_filtered, bins=50)
                mode_index = np.argmax(hist)
                baseline_mode = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
                baseline_values.append(baseline_mode)
            except:
                # Если не удалось найти моду, используем медиану
                baseline_values.append(np.median(window_filtered))
            
            baseline_segments.append({
                'start_time': i / self.sampling_rate,
                'end_time': (i + self.window_size_10min) / self.sampling_rate,
                'baseline': baseline_values[-1],
                'samples_used': len(window_filtered),
                'total_samples': len(window)
            })
        
        # Общая оценка базального ритма
        overall_baseline = np.median(baseline_values) if baseline_values else np.median(fhr_signal)
        
        # Классификация
        if overall_baseline < 110:
            classification = "Брадикардия"
            if overall_baseline < 100:
                severity = "Выраженная"
            else:
                severity = "Умеренная"
        elif overall_baseline > 160:
            classification = "Тахикардия"
            if overall_baseline > 180:
                severity = "Выраженная"
            else:
                severity = "Умеренная"
        else:
            classification = "Норма"
            severity = None
        
        return {
            'overall_baseline': overall_baseline,
            'classification': classification,
            'severity': severity,
            'segments': baseline_segments,
            'stability': np.std(baseline_values) if baseline_values else 0
        }
    
    def calculate_variability(self, fhr_signal, baseline_fhr):
        """
        Расчет вариабельности сердечного ритма
        
        Args:
            fhr_signal (array): Сигнал ЧСС
            baseline_fhr (float): Базальная ЧСС
            
        Returns:
            dict: Анализ вариабельности
        """
        # Кратковременная вариабельность (STV)
        # Расчет как средняя разница между соседними значениями
        rr_intervals = np.diff(60000 / fhr_signal)  # Преобразование в R-R интервалы (мс)
        stv = np.mean(np.abs(np.diff(rr_intervals))) if len(rr_intervals) > 1 else 0
        
        # Долговременная вариабельность (LTV)
        ltv_values = []
        ltv_frequencies = []
        
        # Анализ по минутным сегментам
        for i in range(0, len(fhr_signal) - self.window_size_1min, self.window_size_1min):
            window = fhr_signal[i:i + self.window_size_1min]
            
            # Амплитуда вариабельности
            amplitude = np.max(window) - np.min(window)
            ltv_values.append(amplitude)
            
            # Частота осцилляций (количество пересечений базального уровня)
            crossings = np.sum(np.diff(np.sign(window - baseline_fhr)) != 0)
            frequency = crossings / 2  # Количество полных циклов
            ltv_frequencies.append(frequency)
        
        # Средние значения
        ltv_amplitude = np.mean(ltv_values) if ltv_values else 0
        ltv_frequency = np.mean(ltv_frequencies) if ltv_frequencies else 0
        
        # Классификация типа кривой
        if ltv_amplitude > 25:
            curve_type = "Сальтаторная"
        elif ltv_amplitude < 5:
            # Проверяем длительность монотонности
            monotonic_duration = self._calculate_monotonic_duration(fhr_signal, baseline_fhr)
            if monotonic_duration > 50 * 60 * self.sampling_rate:  # >50 минут
                curve_type = "Монотонная"
            else:
                curve_type = "Сниженная вариабельность"
        elif self._detect_sinusoidal_pattern(fhr_signal, baseline_fhr):
            curve_type = "Синусоидальная"
        else:
            curve_type = "Нормальная (ундирующая)"
        
        return {
            'short_term_variation': stv,
            'long_term_amplitude': ltv_amplitude,
            'long_term_frequency': ltv_frequency,
            'curve_type': curve_type,
            'minute_values': ltv_values,
            'stv_classification': 'Норма' if stv > 3.0 else 'Снижена' if stv > 2.5 else 'Критически снижена'
        }
    
    def _calculate_monotonic_duration(self, fhr_signal, baseline_fhr):
        """Расчет длительности монотонных участков"""
        # Определяем участки с низкой вариабельностью
        window_size = self.window_size_1min
        monotonic_samples = 0
        
        for i in range(0, len(fhr_signal) - window_size, window_size):
            window = fhr_signal[i:i + window_size]
            amplitude = np.max(window) - np.min(window)
            if amplitude < 5:
                monotonic_samples += window_size
        
        return monotonic_samples
    
    def _detect_sinusoidal_pattern(self, fhr_signal, baseline_fhr):
        """Детекция синусоидального паттерна"""
        # Анализ частотного спектра для выявления доминирующей частоты
        freqs, psd = signal.welch(fhr_signal - baseline_fhr, fs=self.sampling_rate, nperseg=min(len(fhr_signal)//4, 256))
        
        # Ищем пик в диапазоне 2-5 циклов в минуту
        freq_range = (freqs >= 2/60) & (freqs <= 5/60)
        if np.any(freq_range):
            max_power_freq = freqs[freq_range][np.argmax(psd[freq_range])]
            total_power = np.sum(psd)
            peak_power = np.max(psd[freq_range])
            
            # Синусоидальный паттерн: доминирующая частота содержит >50% мощности
            return (peak_power / total_power) > 0.5
        
        return False
    
    def detect_accelerations(self, fhr_signal, baseline_fhr):
        """
        Детекция акцелераций
        
        Args:
            fhr_signal (array): Сигнал ЧСС
            baseline_fhr (float): Базальная ЧСС
            
        Returns:
            list: Список обнаруженных акцелераций
        """
        accelerations = []
        
        # Находим участки выше baseline + 15 уд/мин
        threshold = baseline_fhr + 15
        above_threshold = fhr_signal > threshold
        
        # Находим начала и концы акцелераций
        diff_above = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        starts = np.where(diff_above == 1)[0]
        ends = np.where(diff_above == -1)[0]
        
        for start, end in zip(starts, ends):
            duration = (end - start) / self.sampling_rate
            
            # Проверяем критерии: длительность от 15 сек до 10 мин
            if 15 <= duration <= 600:  # 15 сек - 10 мин
                peak_value = np.max(fhr_signal[start:end])
                amplitude = peak_value - baseline_fhr
                
                accelerations.append({
                    'start_time': start / self.sampling_rate,
                    'end_time': end / self.sampling_rate,
                    'duration': duration,
                    'peak_value': peak_value,
                    'amplitude': amplitude,
                    'start_index': start,
                    'end_index': end
                })
        
        # Оценка реактивности (NST)
        # Считаем акцелерации за 20-40 минутные интервалы
        nst_results = []
        for window_duration in [20, 40]:  # минуты
            window_size = int(window_duration * 60 * self.sampling_rate)
            for i in range(0, len(fhr_signal) - window_size, window_size):
                window_start_time = i / self.sampling_rate
                window_end_time = (i + window_size) / self.sampling_rate
                
                # Считаем акцелерации в этом окне
                window_accelerations = [acc for acc in accelerations 
                                      if window_start_time <= acc['start_time'] < window_end_time]
                
                is_reactive = len(window_accelerations) >= 2
                nst_results.append({
                    'window_duration': window_duration,
                    'start_time': window_start_time,
                    'end_time': window_end_time,
                    'accelerations_count': len(window_accelerations),
                    'is_reactive': is_reactive
                })
        
        return {
            'accelerations': accelerations,
            'total_count': len(accelerations),
            'nst_results': nst_results,
            'overall_reactivity': any(result['is_reactive'] for result in nst_results)
        }
    
    def detect_decelerations(self, fhr_signal, uc_signal, baseline_fhr):
        """
        Детекция и классификация децелераций
        
        Args:
            fhr_signal (array): Сигнал ЧСС
            uc_signal (array): Сигнал маточных сокращений (может быть None)
            baseline_fhr (float): Базальная ЧСС
            
        Returns:
            dict: Информация о децелерациях
        """
        decelerations = []
        
        # Находим участки ниже baseline - 15 уд/мин
        threshold = baseline_fhr - 15
        below_threshold = fhr_signal < threshold
        
        # Находим начала и концы децелераций
        diff_below = np.diff(np.concatenate(([False], below_threshold, [False])).astype(int))
        starts = np.where(diff_below == 1)[0]
        ends = np.where(diff_below == -1)[0]
        
        # Детекция схваток если есть UC сигнал
        contractions = []
        if uc_signal is not None:
            contractions = self._detect_contractions(uc_signal)
        
        for start, end in zip(starts, ends):
            duration = (end - start) / self.sampling_rate
            
            # Проверяем критерии: длительность от 15 сек до 10 мин
            if 15 <= duration <= 600:  # 15 сек - 10 мин
                nadir_value = np.min(fhr_signal[start:end])
                amplitude = baseline_fhr - nadir_value
                
                # Классификация типа децелерации
                decel_type = self._classify_deceleration(
                    start, end, fhr_signal, contractions, baseline_fhr
                )
                
                # Особая проверка для пролонгированных
                if duration > 180:  # >3 минут
                    decel_type = "Пролонгированная"
                
                decelerations.append({
                    'start_time': start / self.sampling_rate,
                    'end_time': end / self.sampling_rate,
                    'duration': duration,
                    'nadir_value': nadir_value,
                    'amplitude': amplitude,
                    'type': decel_type,
                    'start_index': start,
                    'end_index': end
                })
        
        # Анализ по типам
        type_counts = {}
        for decel in decelerations:
            decel_type = decel['type']
            if decel_type not in type_counts:
                type_counts[decel_type] = 0
            type_counts[decel_type] += 1
        
        return {
            'decelerations': decelerations,
            'total_count': len(decelerations),
            'type_counts': type_counts,
            'has_late_decelerations': 'Поздняя' in type_counts,
            'has_prolonged_decelerations': 'Пролонгированная' in type_counts
        }
    
    def _detect_contractions(self, uc_signal):
        """Детекция маточных сокращений"""
        contractions = []
        
        # Простая детекция по порогу
        baseline_uc = np.median(uc_signal)
        threshold = baseline_uc + np.std(uc_signal) * 2
        
        above_threshold = uc_signal > threshold
        diff_above = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        starts = np.where(diff_above == 1)[0]
        ends = np.where(diff_above == -1)[0]
        
        for start, end in zip(starts, ends):
            duration = (end - start) / self.sampling_rate
            if duration > 30:  # Минимум 30 секунд для схватки
                peak_time = start + np.argmax(uc_signal[start:end])
                contractions.append({
                    'start_time': start / self.sampling_rate,
                    'peak_time': peak_time / self.sampling_rate,
                    'end_time': end / self.sampling_rate,
                    'duration': duration,
                    'amplitude': np.max(uc_signal[start:end]) - baseline_uc
                })
        
        return contractions
    
    def _classify_deceleration(self, start, end, fhr_signal, contractions, baseline_fhr):
        """Классификация типа децелерации"""
        if not contractions:
            return "Вариабельная"  # Без данных о схватках предполагаем вариабельную
        
        decel_start_time = start / self.sampling_rate
        decel_end_time = end / self.sampling_rate
        decel_nadir_time = start + np.argmin(fhr_signal[start:end])
        decel_nadir_time /= self.sampling_rate
        
        # Находим ближайшую схватку
        closest_contraction = None
        min_distance = float('inf')
        
        for contraction in contractions:
            # Расстояние между началом децелерации и началом схватки
            distance = abs(decel_start_time - contraction['start_time'])
            if distance < min_distance:
                min_distance = distance
                closest_contraction = contraction
        
        if closest_contraction is None or min_distance > 120:  # >2 минут
            return "Вариабельная"
        
        # Анализ временной связи
        contraction_start = closest_contraction['start_time']
        contraction_peak = closest_contraction['peak_time']
        contraction_end = closest_contraction['end_time']
        
        # Ранние децелерации: синхронные со схваткой
        if (abs(decel_start_time - contraction_start) < 30 and  # Начинаются вместе
            abs(decel_nadir_time - contraction_peak) < 30):     # Пик совпадает с пиком схватки
            return "Ранняя"
        
        # Поздние децелерации: запаздывают относительно схватки
        if (decel_start_time > contraction_start + 20 and      # Начинается с задержкой >20 сек
            decel_nadir_time > contraction_peak):              # Пик после пика схватки
            return "Поздняя"
        
        # Все остальные - вариабельные
        return "Вариабельная"
    
    def analyze_uterine_activity(self, uc_signal):
        """
        Анализ маточной активности
        
        Args:
            uc_signal (array): Сигнал маточных сокращений
            
        Returns:
            dict: Анализ маточной активности
        """
        if uc_signal is None:
            return {'status': 'No UC signal provided'}
        
        contractions = self._detect_contractions(uc_signal)
        
        # Анализ по 10-минутным интервалам
        analysis_results = []
        window_size = self.window_size_10min
        
        for i in range(0, len(uc_signal) - window_size, window_size):
            window_start_time = i / self.sampling_rate
            window_end_time = (i + window_size) / self.sampling_rate
            
            # Схватки в этом окне
            window_contractions = [c for c in contractions 
                                 if window_start_time <= c['start_time'] < window_end_time]
            
            frequency = len(window_contractions)  # Количество схваток за 10 мин
            avg_duration = np.mean([c['duration'] for c in window_contractions]) if window_contractions else 0
            avg_amplitude = np.mean([c['amplitude'] for c in window_contractions]) if window_contractions else 0
            
            # Тонус матки (базальный уровень)
            window_signal = uc_signal[i:i + window_size]
            baseline_tone = np.percentile(window_signal, 10)  # 10-й процентиль как базальный тонус
            
            analysis_results.append({
                'start_time': window_start_time,
                'end_time': window_end_time,
                'frequency': frequency,
                'avg_duration': avg_duration,
                'avg_amplitude': avg_amplitude,
                'baseline_tone': baseline_tone,
                'is_normal_frequency': frequency <= 5  # Норма ≤5 схваток за 10 мин
            })
        
        # Общая оценка
        total_contractions = len(contractions)
        avg_frequency = np.mean([r['frequency'] for r in analysis_results]) if analysis_results else 0
        avg_duration_overall = np.mean([c['duration'] for c in contractions]) if contractions else 0
        avg_amplitude_overall = np.mean([c['amplitude'] for c in contractions]) if contractions else 0
        
        return {
            'contractions': contractions,
            'total_contractions': total_contractions,
            'average_frequency_per_10min': avg_frequency,
            'average_duration': avg_duration_overall,
            'average_amplitude': avg_amplitude_overall,
            'analysis_by_windows': analysis_results,
            'is_tachysystole': avg_frequency > 5  # Тахисистолия
        }
    
    def calculate_fisher_score(self, baseline_fhr, variability, accelerations, decelerations):
        """
        Расчет оценки по шкале Фишера
        
        Returns:
            dict: Оценка по Фишеру с детализацией
        """
        score = 0
        details = {}
        
        # 1. Базальная ЧСС (0-2 балла)
        if 110 <= baseline_fhr <= 160:
            fhr_score = 2
        elif (100 <= baseline_fhr < 110) or (160 < baseline_fhr <= 180):
            fhr_score = 1
        else:
            fhr_score = 0
        
        score += fhr_score
        details['baseline_fhr'] = {'value': baseline_fhr, 'score': fhr_score}
        
        # 2. Вариабельность (0-2 балла)
        ltv_amplitude = variability['long_term_amplitude']
        if 5 <= ltv_amplitude <= 25:
            var_score = 2
        elif (3 <= ltv_amplitude < 5) or (25 < ltv_amplitude <= 40):
            var_score = 1
        else:
            var_score = 0
        
        score += var_score
        details['variability'] = {'amplitude': ltv_amplitude, 'score': var_score}
        
        # 3. Акцелерации (0-2 балла)
        acc_count = accelerations['total_count']
        recording_duration = 40  # Предполагаем 40-минутную запись
        
        if acc_count >= 2:
            acc_score = 2
        elif acc_count == 1:
            acc_score = 1
        else:
            acc_score = 0
        
        score += acc_score
        details['accelerations'] = {'count': acc_count, 'score': acc_score}
        
        # 4. Децелерации (0-2 балла)
        has_late = decelerations.get('has_late_decelerations', False)
        has_prolonged = decelerations.get('has_prolonged_decelerations', False)
        variable_count = decelerations['type_counts'].get('Вариабельная', 0)
        
        if not has_late and not has_prolonged and variable_count == 0:
            decel_score = 2  # Нет децелераций
        elif not has_late and not has_prolonged and variable_count <= 2:
            decel_score = 1  # Единичные вариабельные
        else:
            decel_score = 0  # Есть патологические децелерации
        
        score += decel_score
        details['decelerations'] = {
            'late': has_late,
            'prolonged': has_prolonged,
            'variable_count': variable_count,
            'score': decel_score
        }
        
        # Интерпретация
        if score >= 8:
            interpretation = "Нормальное состояние"
            risk_level = "Низкий"
        elif 5 <= score <= 7:
            interpretation = "Пограничное состояние"
            risk_level = "Средний"
        else:
            interpretation = "Патологическое состояние"
            risk_level = "Высокий"
        
        return {
            'total_score': score,
            'max_score': 8,
            'interpretation': interpretation,
            'risk_level': risk_level,
            'details': details
        }
    
    def classify_fetal_condition(self, baseline_fhr_data, variability, accelerations, decelerations, fisher_score):
        """
        Комплексная классификация состояния плода
        
        Returns:
            dict: Классификация состояния плода
        """
        conditions = []
        recommendations = []
        
        baseline_fhr = baseline_fhr_data['overall_baseline']
        ltv_amplitude = variability['long_term_amplitude']
        stv = variability['short_term_variation']
        
        # 1. Оценка базального ритма
        if baseline_fhr_data['classification'] == "Тахикардия":
            # Проверяем компенсацию
            if (ltv_amplitude >= 5 and 
                accelerations['total_count'] >= 1 and 
                not decelerations.get('has_late_decelerations', False)):
                conditions.append("Компенсированная тахикардия")
                recommendations.append("Мониторинг, поиск причин (лихорадка, инфекция)")
            else:
                conditions.append("Декомпенсированная тахикардия")
                recommendations.append("Признак гипоксии - требуется экстренное вмешательство")
        
        elif baseline_fhr_data['classification'] == "Брадикардия":
            if (ltv_amplitude >= 5 and 
                accelerations['total_count'] >= 1):
                conditions.append("Компенсированная брадикардия")
                recommendations.append("Возможный вариант нормы, требует наблюдения")
            else:
                conditions.append("Декомпенсированная брадикардия")
                recommendations.append("Признак тяжелой хронической гипоксии - экстренное родоразрешение")
        
        # 2. Оценка гипоксии по стадиям
        if fisher_score['total_score'] <= 4:
            if (ltv_amplitude < 2 and 
                stv < 2.5 and 
                accelerations['total_count'] == 0):
                conditions.append("Декомпенсированная гипоксия (стадия 3)")
                recommendations.append("НЕМЕДЛЕННОЕ РОДОРАЗРЕШЕНИЕ")
            else:
                conditions.append("Выраженная гипоксия (стадия 2)")
                recommendations.append("Экстренное родоразрешение")
        
        elif 5 <= fisher_score['total_score'] <= 7:
            conditions.append("Начальные явления гипоксии (стадия 1)")
            recommendations.append("Углубленное обследование, лечение, решение о родоразрешении")
        
        # 3. Проверка на синусоидальный ритм
        if variability['curve_type'] == "Синусоидальная":
            conditions.append("Синусоидальный ритм")
            recommendations.append("КРАЙНЕ НЕБЛАГОПРИЯТНЫЙ ПРИЗНАК - немедленное родоразрешение")
        
        # 4. Ареактивный НСТ
        if not accelerations.get('overall_reactivity', True):
            conditions.append("Ареактивный нестрессовый тест")
            recommendations.append("Требует дифференциальной диагностики, повторный тест или БПП")
        
        # 5. Общая классификация по FIGO
        if fisher_score['total_score'] >= 8:
            figo_class = "Нормальный"
        elif 5 <= fisher_score['total_score'] <= 7:
            figo_class = "Сомнительный"
        else:
            figo_class = "Патологический"
        
        # Если нет специфических состояний и оценка хорошая
        if not conditions and fisher_score['total_score'] >= 8:
            conditions.append("Нормальное состояние плода")
            recommendations.append("Продолжение мониторинга в обычном режиме")
        
        return {
            'primary_conditions': conditions,
            'figo_classification': figo_class,
            'fisher_score': fisher_score['total_score'],
            'risk_level': fisher_score['risk_level'],
            'recommendations': recommendations,
            'requires_immediate_delivery': any('немедленное' in rec.lower() or 'экстренное' in rec.lower() 
                                             for rec in recommendations)
        }
    
    def comprehensive_analysis(self, fhr_signal, uc_signal=None):
        """
        Комплексный анализ сигнала КТГ
        
        Args:
            fhr_signal (array): Сигнал ЧСС плода
            uc_signal (array): Сигнал маточных сокращений (опционально)
            
        Returns:
            dict: Полный анализ КТГ
        """
        # Предобработка
        preprocessed = self.preprocess_signal(fhr_signal, uc_signal)
        fhr_clean = preprocessed['fhr_smoothed']
        uc_clean = preprocessed['uc_clean']
        
        # Детекция артефактов
        artifacts = self.detect_artifacts(fhr_clean)
        
        # Базальная ЧСС
        baseline_fhr_data = self.calculate_baseline_fhr(fhr_clean)
        baseline_fhr = baseline_fhr_data['overall_baseline']
        
        # Вариабельность
        variability = self.calculate_variability(fhr_clean, baseline_fhr)
        
        # Акцелерации
        accelerations = self.detect_accelerations(fhr_clean, baseline_fhr)
        
        # Децелерации
        decelerations = self.detect_decelerations(fhr_clean, uc_clean, baseline_fhr)
        
        # Маточная активность
        uterine_activity = self.analyze_uterine_activity(uc_clean)
        
        # Оценка по Фишеру
        fisher_score = self.calculate_fisher_score(
            baseline_fhr, variability, accelerations, decelerations
        )
        
        # Классификация состояния плода
        fetal_condition = self.classify_fetal_condition(
            baseline_fhr_data, variability, accelerations, decelerations, fisher_score
        )
        
        return {
            'signal_quality': {
                'artifacts': artifacts,
                'duration_minutes': len(fhr_signal) / (self.sampling_rate * 60),
                'quality_score': 100 - artifacts['total_artifact_percentage']
            },
            'baseline_fhr': baseline_fhr_data,
            'variability': variability,
            'accelerations': accelerations,
            'decelerations': decelerations,
            'uterine_activity': uterine_activity,
            'fisher_score': fisher_score,
            'fetal_condition': fetal_condition,
            'metadata': {
                'sampling_rate': self.sampling_rate,
                'analysis_timestamp': pd.Timestamp.now(),
                'signal_length': len(fhr_signal)
            }
        }


# Пример использования и тестирования
if __name__ == "__main__":
    # Создание тестовых данных
    np.random.seed(42)
    duration_minutes = 30
    sampling_rate = 4.0
    n_samples = int(duration_minutes * 60 * sampling_rate)
    
    # Симуляция ЧСС плода с различными паттернами
    time_axis = np.linspace(0, duration_minutes * 60, n_samples)
    
    # Базальный ритм
    baseline = 140
    
    # Долговременная вариабельность
    ltv = 10 * np.sin(2 * np.pi * time_axis / 60)  # Цикл 1 минута
    
    # Кратковременная вариабельность
    stv = 2 * np.random.randn(n_samples)
    
    # Тренд (может показывать брадикардию в конце)
    trend = -5 * (time_axis / (duration_minutes * 60))
    
    # Акцелерации (случайные)
    accelerations_mask = np.random.random(n_samples) < 0.02  # 2% точек
    acceleration_signal = np.zeros(n_samples)
    for i in range(len(accelerations_mask)):
        if accelerations_mask[i]:
            # Создаем акцелерацию длительностью ~30 сек
            acc_duration = int(30 * sampling_rate)
            end_idx = min(i + acc_duration, n_samples)
            acceleration_signal[i:end_idx] = 20 * np.exp(-np.arange(end_idx - i) / (10 * sampling_rate))
    
    # Итоговый сигнал ЧСС
    fhr_signal = baseline + ltv + stv + trend + acceleration_signal
    
    # Добавляем артефакты
    # Потеря сигнала
    artifact_start = int(10 * 60 * sampling_rate)  # 10 минута
    artifact_end = int(12 * 60 * sampling_rate)    # 12 минута
    fhr_signal[artifact_start:artifact_end] = 0
    
    # Симуляция маточной активности
    uc_signal = np.zeros(n_samples)
    # Добавляем схватки каждые 3-4 минуты
    for contraction_time in range(0, duration_minutes, 4):
        start_idx = int(contraction_time * 60 * sampling_rate)
        if start_idx < n_samples:
            contraction_duration = int(60 * sampling_rate)  # 1 минута
            end_idx = min(start_idx + contraction_duration, n_samples)
            # Колоколообразная форма схватки
            contraction_shape = 50 * np.exp(-((np.arange(end_idx - start_idx) - contraction_duration//2) ** 2) / (2 * (contraction_duration//6) ** 2))
            uc_signal[start_idx:end_idx] += contraction_shape
    
    # Инициализация анализатора
    analyzer = CTGAnalyzer(sampling_rate=sampling_rate)
    
    # Проведение анализа
    print("Проведение комплексного анализа КТГ...")
    results = analyzer.comprehensive_analysis(fhr_signal, uc_signal)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА КТГ")
    print("="*60)
    
    print(f"\n1. КАЧЕСТВО СИГНАЛА:")
    quality = results['signal_quality']
    print(f"   Длительность записи: {quality['duration_minutes']:.1f} мин")
    print(f"   Качество сигнала: {quality['quality_score']:.1f}%")
    print(f"   Процент артефактов: {quality['artifacts']['total_artifact_percentage']:.1f}%")
    
    print(f"\n2. БАЗАЛЬНАЯ ЧСС:")
    baseline = results['baseline_fhr']
    print(f"   Значение: {baseline['overall_baseline']:.1f} уд/мин")
    print(f"   Классификация: {baseline['classification']}")
    if baseline['severity']:
        print(f"   Степень: {baseline['severity']}")
    
    print(f"\n3. ВАРИАБЕЛЬНОСТЬ:")
    variability = results['variability']
    print(f"   Кратковременная (STV): {variability['short_term_variation']:.1f} мс")
    print(f"   Долговременная амплитуда: {variability['long_term_amplitude']:.1f} уд/мин")
    print(f"   Тип кривой: {variability['curve_type']}")
    
    print(f"\n4. АКЦЕЛЕРАЦИИ:")
    accelerations = results['accelerations']
    print(f"   Количество: {accelerations['total_count']}")
    print(f"   Реактивность: {'Да' if accelerations['overall_reactivity'] else 'Нет'}")
    
    print(f"\n5. ДЕЦЕЛЕРАЦИИ:")
    decelerations = results['decelerations']
    print(f"   Общее количество: {decelerations['total_count']}")
    for decel_type, count in decelerations['type_counts'].items():
        print(f"   {decel_type}: {count}")
    
    print(f"\n6. ОЦЕНКА ПО ФИШЕРУ:")
    fisher = results['fisher_score']
    print(f"   Баллы: {fisher['total_score']}/{fisher['max_score']}")
    print(f"   Интерпретация: {fisher['interpretation']}")
    print(f"   Уровень риска: {fisher['risk_level']}")
    
    print(f"\n7. СОСТОЯНИЕ ПЛОДА:")
    condition = results['fetal_condition']
    print(f"   Классификация FIGO: {condition['figo_classification']}")
    print(f"   Основные состояния:")
    for cond in condition['primary_conditions']:
        print(f"     - {cond}")
    print(f"   Рекомендации:")
    for rec in condition['recommendations']:
        print(f"     - {rec}")
    
    if condition['requires_immediate_delivery']:
        print(f"\n⚠️  ТРЕБУЕТСЯ ЭКСТРЕННОЕ ВМЕШАТЕЛЬСТВО!")
    
    print("\n" + "="*60)
