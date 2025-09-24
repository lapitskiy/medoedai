"""
Утилита для загрузки конфигурации из JSON файла
"""
import json
import os
from typing import Any, Optional

_config_cache = None

def load_config(config_path: str = "env.json") -> dict:
    """
    Загружает конфигурацию из JSON файла с кэшированием
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        dict: Словарь с конфигурацией
    """
    global _config_cache
    
    if _config_cache is None:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                _config_cache = json.load(f)
        except FileNotFoundError:
            print(f"Файл конфигурации {config_path} не найден, используется пустой конфиг")
            _config_cache = {}
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON файла {config_path}: {e}")
            _config_cache = {}
    
    return _config_cache

def get_config_value(key: str, default: Any = None, config_path: str = "env.json") -> Any:
    """
    Получает значение из конфигурации
    
    Args:
        key: Ключ конфигурации
        default: Значение по умолчанию
        config_path: Путь к файлу конфигурации
        
    Returns:
        Значение из конфигурации или default
    """
    config = load_config(config_path)
    return config.get(key, default)

def reload_config():
    """
    Принудительная перезагрузка конфигурации
    """
    global _config_cache
    _config_cache = None
