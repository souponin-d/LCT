"""
Модуль для обработки данных КТГ из папки data и интеграции с анализатором
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from ctg_analysis import CTGAnalyzer
import warnings
warnings.filterwarnings('ignore')

class CTGDataProcessor:
    """
    Процессор для массовой обработки данных КТГ
    """
    
    def __init__(self, data_path='./data', sampling_rate=4.0):
        """
        Инициализация процессора
        
        Args:
            data_path (str): Путь к папке с данными
            sampling_rate (float): Частота дискретизации
        """
        self.data_path = Path(data_path)
        self.analyzer = CTGAnalyzer(sampling_rate=sampling_rate)
        self.results_cache = {}
        
    def load_patient_data(self, group_type, patient_id, max_files_per_type=None):
        """
        Загрузка данных пациента
        
        Args:
            group_type (str): 'hypoxia' или 'regular'
            patient_id (str): ID пациента
            max_files_per_type (int): Максимум файлов каждого типа
            
        Returns:
            dict: Загруженные данные
        """
        patient_path = self.data_path / group_type / str(patient_id)
        data = {'bpm': [], 'uterus': [], 'metadata': []}
        
        if not patient_path.exists():
            return data
        
        # Загрузка BPM данных (сигнал ЧСС)
        bpm_path = patient_path / 'bpm'
        if bmp_path.exists():
            csv_files = list(bpm_path.glob('*.csv'))
            if max_files_per_type:
                csv_files = csv_files[:max_files_per_type]
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'time_sec' in df.columns and 'value' in df.columns:
                        data['bpm'].append({
                            'file': csv_file.name,
                            'time': df['time_sec'].values,
                            'fhr': df['value'].values,
                            'duration': df['time_sec'].max() - df['time_sec'].min()
                        })
                except Exception as e:
                    print(f"Ошибка при загрузке {csv_file}: {e}")
        
        # Загрузка Uterus данных (маточная активность)
        uterus_path = patient_path / 'uterus'
        if uterus_path.exists():
            csv_files = list(uterus_path.glob('*.csv'))
            if max_files_per_type:
                csv_files = csv_files[:max_files_per_type]
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'time_sec' in df.columns and 'value' in df.columns:
                        data['uterus'].append({
                            'file': csv_file.name,
                            'time': df['time_sec'].values,
                            'uc': df['value'].values,
                            'duration': df['time_sec'].max() - df['time_sec'].min()
                        })
                except Exception as e:
                    print(f"Ошибка при загрузке {csv_file}: {e}")
        
        # Сопоставление BPM и Uterus файлов
        for bpm_data in data['bpm']:
            # Ищем соответствующий uterus файл
            bmp_file = bpm_data['file']
            expected_uterus_file = bmp_file.replace('_1.', '_2.')
            
            matching_uterus = None
            for uterus_data in data['uterus']:
                if uterus_data['file'] == expected_uterus_file:
                    matching_uterus = uterus_data
                    break
            
            data['metadata'].append({
                'bpm_file': bpm_data['file'],
                'uterus_file': matching_uterus['file'] if matching_uterus else None,
                'has_paired_data': matching_uterus is not None,
                'duration': bpm_data['duration'],
                'group': group_type,
                'patient_id': patient_id
            })
        
        return data
    
    def synchronize_signals(self, fhr_time, fhr_values, uc_time=None, uc_values=None, target_rate=4.0):
        """
        Синхронизация и ресамплинг сигналов
        
        Args:
            fhr_time, fhr_values: Временная ось и значения ЧСС
            uc_time, uc_values: Временная ось и значения UC (опционально)
            target_rate: Целевая частота дискретизации
            
        Returns:
            dict: Синхронизированные сигналы
        """
        # Определяем общий временной интервал
        start_time = fhr_time[0]
        end_time = fhr_time[-1]
        
        if uc_time is not None:
            start_time = max(start_time, uc_time[0])
            end_time = min(end_time, uc_time[-1])
        
        # Создаем равномерную временную сетку
        n_samples = int((end_time - start_time) * target_rate)
        uniform_time = np.linspace(start_time, end_time, n_samples)
        
        # Интерполяция FHR
        fhr_resampled = np.interp(uniform_time, fhr_time, fhr_values)
        
        # Интерполяция UC если есть
        uc_resampled = None
        if uc_time is not None and uc_values is not None:
            uc_resampled = np.interp(uniform_time, uc_time, uc_values)
        
        return {
            'time': uniform_time,
            'fhr': fhr_resampled,
            'uc': uc_resampled,
            'sampling_rate': target_rate,
            'duration': end_time - start_time
        }
    
    def analyze_patient(self, group_type, patient_id, max_files=3):
        """
        Анализ данных одного пациента
        
        Args:
            group_type (str): Тип группы
            patient_id (str): ID пациента
            max_files (int): Максимум файлов для анализа
            
        Returns:
            dict: Результаты анализа пациента
        """
        # Загрузка данных
        patient_data = self.load_patient_data(group_type, patient_id, max_files)
        
        if not patient_data['bpm']:
            return {'status': 'No data found', 'patient_id': patient_id, 'group': group_type}
        
        # Анализ каждого файла
        file_results = []
        
        for i, bpm_data in enumerate(patient_data['bpm']):
            # Находим соответствующий UC файл
            matching_uc = None
            for uc_data in patient_data['uterus']:
                if uc_data['file'] == bmp_data['file'].replace('_1.', '_2.'):
                    matching_uc = uc_data
                    break
            
            # Синхронизация сигналов
            if matching_uc:
                synced = self.synchronize_signals(
                    bpm_data['time'], bmp_data['fhr'],
                    matching_uc['time'], matching_uc['uc']
                )
            else:
                synced = self.synchronize_signals(
                    bpm_data['time'], bmp_data['fhr']
                )
            
            # Проведение анализа
            try:
                analysis = self.analyzer.comprehensive_analysis(
                    synced['fhr'], synced['uc']
                )
                
                analysis['file_info'] = {
                    'bpm_file': bmp_data['file'],
                    'uc_file': matching_uc['file'] if matching_uc else None,
                    'duration': synced['duration'],
                    'has_uc_data': matching_uc is not None
                }
                
                file_results.append(analysis)
                
            except Exception as e:
                print(f"Ошибка анализа файла {bmp_data['file']}: {e}")
                continue
        
        # Агрегация результатов по пациенту
        if file_results:
            patient_summary = self._aggregate_patient_results(file_results, group_type, patient_id)
        else:
            patient_summary = {'status': 'Analysis failed', 'patient_id': patient_id, 'group': group_type}
        
        return {
            'patient_summary': patient_summary,
            'file_results': file_results,
            'patient_id': patient_id,
            'group': group_type
        }
    
    def _aggregate_patient_results(self, file_results, group_type, patient_id):
        """Агрегация результатов анализа по пациенту"""
        
        # Собираем ключевые метрики
        baseline_values = []
        fisher_scores = []
        variability_scores = []
        acceleration_counts = []
        deceleration_counts = []
        pathological_count = 0
        
        for result in file_results:
            baseline_values.append(result['baseline_fhr']['overall_baseline'])
            fisher_scores.append(result['fisher_score']['total_score'])
            variability_scores.append(result['variability']['long_term_amplitude'])
            acceleration_counts.append(result['accelerations']['total_count'])
            deceleration_counts.append(result['decelerations']['total_count'])
            
            if result['fetal_condition']['figo_classification'] == 'Патологический':
                pathological_count += 1
        
        # Определяем общий риск пациента
        avg_fisher = np.mean(fisher_scores)
        if avg_fisher >= 8:
            overall_risk = "Низкий"
        elif avg_fisher >= 5:
            overall_risk = "Средний"
        else:
            overall_risk = "Высокий"
        
        # Проверяем согласованность с группой
        expected_high_risk = (group_type == 'hypoxia')
        risk_matches_group = (overall_risk == "Высокий") == expected_high_risk
        
        return {
            'patient_id': patient_id,
            'group': group_type,
            'files_analyzed': len(file_results),
            'average_baseline_fhr': np.mean(baseline_values),
            'average_fisher_score': avg_fisher,
            'average_variability': np.mean(variability_scores),
            'total_accelerations': np.sum(acceleration_counts),
            'total_decelerations': np.sum(deceleration_counts),
            'pathological_files_count': pathological_count,
            'pathological_percentage': (pathological_count / len(file_results)) * 100,
            'overall_risk_level': overall_risk,
            'risk_matches_group': risk_matches_group,
            'requires_attention': pathological_count > 0 or avg_fisher < 6
        }
    
    def batch_analyze(self, group_types=['hypoxia', 'regular'], max_patients_per_group=10, max_files_per_patient=2):
        """
        Массовый анализ данных
        
        Args:
            group_types (list): Типы групп для анализа
            max_patients_per_group (int): Максимум пациентов в группе
            max_files_per_patient (int): Максимум файлов на пациента
            
        Returns:
            dict: Результаты массового анализа
        """
        all_results = {}
        
        for group_type in group_types:
            print(f"\nАнализ группы: {group_type}")
            group_path = self.data_path / group_type
            
            if not group_path.exists():
                print(f"Папка {group_path} не найдена")
                continue
            
            # Получаем список пациентов
            patient_dirs = [d for d in group_path.iterdir() 
                          if d.is_dir() and d.name.isdigit()]
            patient_dirs = sorted(patient_dirs, key=lambda x: int(x.name))[:max_patients_per_group]
            
            group_results = []
            
            for patient_dir in tqdm(patient_dirs, desc=f"Обработка {group_type}"):
                patient_id = patient_dir.name
                
                try:
                    patient_result = self.analyze_patient(
                        group_type, patient_id, max_files_per_patient
                    )
                    
                    if 'patient_summary' in patient_result:
                        group_results.append(patient_result)
                        
                except Exception as e:
                    print(f"Ошибка анализа пациента {patient_id}: {e}")
                    continue
            
            all_results[group_type] = group_results
        
        # Создаем сводную статистику
        summary_stats = self._create_summary_statistics(all_results)
        
        return {
            'results_by_group': all_results,
            'summary_statistics': summary_stats,
            'analysis_metadata': {
                'total_patients_analyzed': sum(len(results) for results in all_results.values()),
                'groups_analyzed': list(all_results.keys()),
                'max_patients_per_group': max_patients_per_group,
                'max_files_per_patient': max_files_per_patient
            }
        }
    
    def _create_summary_statistics(self, all_results):
        """Создание сводной статистики"""
        summary = {}
        
        for group_type, group_results in all_results.items():
            if not group_results:
                continue
            
            # Извлекаем метрики
            patient_summaries = [r['patient_summary'] for r in group_results 
                               if 'patient_summary' in r and 'average_fisher_score' in r['patient_summary']]
            
            if not patient_summaries:
                continue
            
            fisher_scores = [p['average_fisher_score'] for p in patient_summaries]
            baseline_fhrs = [p['average_baseline_fhr'] for p in patient_summaries]
            variabilities = [p['average_variability'] for p in patient_summaries]
            pathological_percentages = [p['pathological_percentage'] for p in patient_summaries]
            
            high_risk_count = sum(1 for p in patient_summaries if p['overall_risk_level'] == 'Высокий')
            
            summary[group_type] = {
                'patients_count': len(patient_summaries),
                'average_fisher_score': np.mean(fisher_scores),
                'std_fisher_score': np.std(fisher_scores),
                'average_baseline_fhr': np.mean(baseline_fhrs),
                'std_baseline_fhr': np.std(baseline_fhrs),
                'average_variability': np.mean(variabilities),
                'std_variability': np.std(variabilities),
                'high_risk_patients': high_risk_count,
                'high_risk_percentage': (high_risk_count / len(patient_summaries)) * 100,
                'average_pathological_files': np.mean(pathological_percentages),
                'classification_accuracy': np.mean([p['risk_matches_group'] for p in patient_summaries]) * 100
            }
        
        return summary
    
    def save_results(self, results, output_path='ctg_analysis_results.json'):
        """Сохранение результатов в файл"""
        
        # Преобразуем numpy типы в стандартные Python типы для JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Результаты сохранены в {output_path}")
    
    def create_summary_report(self, results):
        """Создание текстового отчета"""
        
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ПО АНАЛИЗУ ДАННЫХ КТГ")
        report.append("=" * 80)
        
        summary_stats = results.get('summary_statistics', {})
        
        for group_type, stats in summary_stats.items():
            report.append(f"\n📊 ГРУППА: {group_type.upper()}")
            report.append("-" * 50)
            report.append(f"Количество пациентов: {stats['patients_count']}")
            report.append(f"Средняя оценка по Фишеру: {stats['average_fisher_score']:.2f} ± {stats['std_fisher_score']:.2f}")
            report.append(f"Средняя базальная ЧСС: {stats['average_baseline_fhr']:.1f} ± {stats['std_baseline_fhr']:.1f} уд/мин")
            report.append(f"Средняя вариабельность: {stats['average_variability']:.1f} ± {stats['std_variability']:.1f} уд/мин")
            report.append(f"Пациентов высокого риска: {stats['high_risk_patients']} ({stats['high_risk_percentage']:.1f}%)")
            report.append(f"Точность классификации: {stats['classification_accuracy']:.1f}%")
            
            # Интерпретация
            if group_type == 'hypoxia':
                expected_high_risk = True
                interpretation = "ожидается высокий процент пациентов высокого риска"
            else:
                expected_high_risk = False
                interpretation = "ожидается низкий процент пациентов высокого риска"
            
            report.append(f"Ожидание: {interpretation}")
            
            if stats['high_risk_percentage'] > 50 and group_type == 'hypoxia':
                report.append("✅ Результат соответствует ожиданиям для группы гипоксии")
            elif stats['high_risk_percentage'] < 30 and group_type == 'regular':
                report.append("✅ Результат соответствует ожиданиям для контрольной группы")
            else:
                report.append("⚠️  Результат требует дополнительного анализа")
        
        # Сравнение групп
        if len(summary_stats) == 2:
            hypoxia_stats = summary_stats.get('hypoxia', {})
            regular_stats = summary_stats.get('regular', {})
            
            if hypoxia_stats and regular_stats:
                report.append(f"\n🔍 СРАВНЕНИЕ ГРУПП")
                report.append("-" * 50)
                
                fisher_diff = hypoxia_stats['average_fisher_score'] - regular_stats['average_fisher_score']
                report.append(f"Разница в оценке Фишера: {fisher_diff:.2f}")
                
                risk_diff = hypoxia_stats['high_risk_percentage'] - regular_stats['high_risk_percentage']
                report.append(f"Разница в проценте высокого риска: {risk_diff:.1f}%")
                
                if fisher_diff < -1 and risk_diff > 20:
                    report.append("✅ Алгоритм успешно различает группы")
                else:
                    report.append("⚠️  Различия между группами недостаточно выражены")
        
        return "\n".join(report)


# Демонстрационный код
if __name__ == "__main__":
    print("Инициализация процессора КТГ данных...")
    
    # Создаем процессор
    processor = CTGDataProcessor(data_path='./data')
    
    # Проводим анализ небольшой выборки для демонстрации
    print("Проведение анализа данных...")
    results = processor.batch_analyze(
        group_types=['hypoxia', 'regular'],
        max_patients_per_group=3,  # Берем по 3 пациента для демонстрации
        max_files_per_patient=2    # По 2 файла на пациента
    )
    
    # Создаем отчет
    report = processor.create_summary_report(results)
    print("\n" + report)
    
    # Сохраняем результаты
    processor.save_results(results, 'ctg_demo_results.json')
    
    # Дополнительная статистика
    print(f"\n📈 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
    print(f"Всего проанализировано пациентов: {results['analysis_metadata']['total_patients_analyzed']}")
    
    for group_type, group_results in results['results_by_group'].items():
        total_files = sum(len(r.get('file_results', [])) for r in group_results)
        print(f"Файлов проанализировано в группе {group_type}: {total_files}")
    
    print(f"\nРезультаты сохранены в ctg_demo_results.json")
    print(f"Полный анализ завершен!")
