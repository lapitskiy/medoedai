"""
Тренер для обучения CNN моделей на данных криптовалют
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from .models import TradingCNN, MultiTimeframeCNN, PricePredictionCNN
from .data_loader import CryptoDataLoader, CryptoDataset, MultiTimeframeDataset
from .config import CNNTrainingConfig


class CNNTrainer:
    """Тренер для обучения CNN моделей"""
    
    def __init__(self, config: CNNTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Настройка логирования
        self._setup_logging()
        
        # Инициализация компонентов
        self.data_loader = CryptoDataLoader(config)
        self.model = None
        self.classifier_head: Optional[nn.Module] = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.run_id: Optional[str] = None
        
        # Метрики обучения
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        
        # Wandb (если включено)
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    config=config.to_dict(),
                    name=f"cnn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.wandb = wandb
            except ImportError:
                print("⚠️ Wandb не установлен, отключаем логирование")
                self.wandb = None
        else:
            self.wandb = None
    
    def _setup_logging(self):
        """Настройка логирования"""
        # Настройка логирования только в консоль (без файлов)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Только консоль
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_model(self, model_type: str = "single") -> nn.Module:
        """Создание модели в зависимости от типа"""
        if model_type == "single":
            self.model = TradingCNN(
                input_channels=self.config.input_channels,
                hidden_channels=self.config.hidden_channels,
                output_features=self.config.output_features,
                sequence_length=self.config.sequence_length,
                dropout_rate=self.config.dropout_rate,
                use_batch_norm=self.config.use_batch_norm
            ).to(self.device)
        
        elif model_type == "multiframe":
            sequence_lengths = {
                "5m": self.config.sequence_length,
                "15m": self.config.sequence_length // 3,
                "1h": self.config.sequence_length // 12
            }
            
            self.model = MultiTimeframeCNN(
                input_channels=self.config.input_channels,
                hidden_channels=self.config.hidden_channels,
                output_features=self.config.output_features,
                sequence_lengths=sequence_lengths,
                dropout_rate=self.config.dropout_rate,
                use_batch_norm=self.config.use_batch_norm
            ).to(self.device)

            # Классификационная голова для обучения по меткам (3 класса)
            self.classifier_head = nn.Linear(self.config.output_features, getattr(self.config, 'num_classes', 3)).to(self.device)
        
        elif model_type == "prediction":
            self.model = PricePredictionCNN(
                input_channels=self.config.input_channels,
                hidden_channels=self.config.hidden_channels,
                sequence_length=self.config.sequence_length,
                dropout_rate=self.config.dropout_rate,
                use_batch_norm=self.config.use_batch_norm,
                num_classes=self.config.num_classes  # 2 или 3 в зависимости от схемы
            ).to(self.device)
        
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        self.logger.info(f"Создана модель типа {model_type}: {self.model}")
        return self.model
    
    def setup_training(self):
        """Настройка компонентов для обучения"""
        # Оптимизатор
        # Соберем параметры: модель +, при наличии, классификационная голова
        params = list(self.model.parameters())
        if self.classifier_head is not None:
            params += list(self.classifier_head.parameters())
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Неизвестный оптимизатор: {self.config.optimizer}")
        
        # Планировщик
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.num_epochs // 3, gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
        
        # Функция потерь с учетом дисбаланса классов и label smoothing
        label_smoothing = float(getattr(self.config, 'label_smoothing', 0.0))
        if getattr(self.config, 'class_balance', 'auto') == 'auto':
            # По умолчанию слабое смещение против класса 1 при ternary было; теперь адаптивно
            if self.config.num_classes == 2:
                class_weights = torch.tensor([1.0, 1.0], device=self.device)
            else:
                class_weights = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.logger.info(f"Настроен оптимизатор: {self.config.optimizer}")
        self.logger.info(f"Настроен планировщик: {self.config.scheduler}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Перемещаем данные на устройство
            if isinstance(data, dict):
                # Мультифреймовые данные
                data = {k: v.to(self.device) for k, v in data.items()}
            else:
                data = data.to(self.device)
            target = target.to(self.device).squeeze()
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            if isinstance(self.model, MultiTimeframeCNN):
                features = self.model(data)
                output = self.classifier_head(features) if self.classifier_head is not None else features
            else:
                output = self.model(data)
            
            # Вычисляем потери
            loss = self.criterion(output, target)
            
            # Обратный проход
            loss.backward()
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Опционально: weight decay уже в оптимизаторе; применяем step
            self.optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Логирование прогресса
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f'Epoch: {len(self.train_losses)+1}, '
                    f'Batch: {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Acc: {100.*correct/total:.2f}%'
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Перемещаем данные на устройство
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)
                target = target.to(self.device).squeeze()
                
                # Прямой проход
                if isinstance(self.model, MultiTimeframeCNN):
                    features = self.model(data)
                    output = self.classifier_head(features) if self.classifier_head is not None else features
                else:
                    output = self.model(data)
                
                # Вычисляем потери
                loss = self.criterion(output, target)
                
                # Статистика
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, symbol: str, timeframe: str, epoch: int, 
                   val_accuracy: float, is_best: bool = False):
        """Сохранение модели"""
        save_dir = self.config.model_save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Подготавливаем данные для сохранения
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier_head.state_dict() if self.classifier_head is not None else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Сохраняем модель
        # Используем единый run_id на весь запуск, если он задан
        model_path = self.config.get_model_path(symbol, timeframe, run_id=self.run_id)
        torch.save(checkpoint, model_path)
        
        # Сохраняем лучшую модель отдельно
        if is_best:
            best_model_path = model_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"💾 Сохранена лучшая модель: {best_model_path}")
        
        self.logger.info(f"💾 Сохранена модель эпохи {epoch}: {model_path}")
    
    def train_single_model(self, symbol: str, timeframe: str, model_type: str = "prediction"):
        """Обучение модели для одного символа и временного фрейма"""
        self.logger.info(f"🚀 Начинаем обучение {model_type} модели для {symbol} {timeframe}")
        # Генерируем единый run_id для этого запуска, если не задан
        if not self.run_id:
            import time as _time
            self.run_id = hex(int(_time.time()))[-4:]
        
        # Создаем модель
        self.create_model(model_type)
        
        # Настраиваем обучение
        self.setup_training()
        
        # Подготавливаем данные
        data_dict = self.data_loader.prepare_training_data([symbol], [timeframe])
        
        if not data_dict:
            self.logger.error(f"❌ Не удалось загрузить данные для {symbol} {timeframe}")
            return None
        
        # Создаем датасеты
        datasets = self.data_loader.create_datasets(data_dict)
        key = f"{symbol}_{timeframe}"
        
        if key not in datasets:
            self.logger.error(f"❌ Датасет для {key} не найден")
            return None
        
        dataset = datasets[key]
        
        # Разделяем на train/val
        train_size = int(len(dataset) * (1 - self.config.validation_split))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Создаем DataLoader'ы
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"📊 Данные: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Обучение
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_acc = self.validate(val_loader)
            
            # Обновляем планировщик
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Сохраняем метрики
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Проверяем на лучшую модель
            is_best = val_acc > best_val_accuracy
            if is_best:
                best_val_accuracy = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Сохраняем модель
            if epoch % 10 == 0 or is_best:
                self.save_model(symbol, timeframe, epoch, val_acc, is_best)
            
            # Логирование
            epoch_time = time.time() - start_time
            self.logger.info(
                f'Epoch {epoch+1}/{self.config.num_epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                f'Time: {epoch_time:.2f}s'
            )
            
            # Wandb логирование
            if self.wandb:
                self.wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"🛑 Early stopping на эпохе {epoch+1}")
                break
        
        self.logger.info(f"✅ Обучение завершено. Лучшая точность: {best_val_accuracy:.2f}%")
        
        return {
            'model': self.model,
            'best_val_accuracy': best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def train_multiframe_model(self, symbols: List[str]):
        """Обучение мультифреймовой модели для нескольких символов"""
        self.logger.info(f"🚀 Начинаем обучение мультифреймовой модели для {symbols}")
        # Единый run_id для этого запуска
        if not self.run_id:
            import time as _time
            self.run_id = hex(int(_time.time()))[-4:]
        
        # Создаем мультифреймовую модель
        self.create_model("multiframe")
        
        # Настраиваем обучение
        self.setup_training()
        
        # Подготавливаем данные для всех символов и фреймов
        data_dict = self.data_loader.prepare_training_data(symbols, self.config.timeframes)
        
        if not data_dict:
            self.logger.error("❌ Не удалось загрузить данные")
            return None
        
        # Создаем мультифреймовые датасеты
        train_dataset, val_dataset = self.data_loader.create_multiframe_dataset(data_dict)
        
        # Создаем DataLoader'ы
        train_loader, val_loader = self.data_loader.create_dataloaders(train_dataset, val_dataset)
        
        self.logger.info(f"📊 Данные: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Обучение (аналогично train_single_model)
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Обучение и валидация
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Обновляем планировщик
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Сохраняем метрики
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Проверяем на лучшую модель
            is_best = val_acc > best_val_accuracy
            if is_best:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Сохраняем лучшую мультифреймовую модель
                self.save_model("multi", "multi", epoch, val_acc, True)
            else:
                patience_counter += 1
            
            # Логирование
            epoch_time = time.time() - start_time
            self.logger.info(
                f'Epoch {epoch+1}/{self.config.num_epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                f'Time: {epoch_time:.2f}s'
            )
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"🛑 Early stopping на эпохе {epoch+1}")
                break
        
        self.logger.info(f"✅ Обучение мультифреймовой модели завершено. Лучшая точность: {best_val_accuracy:.2f}%")

        # Сохранение manifest и результатов в стиле DQN
        try:
            # Вытягиваем профиль GPU, если есть
            try:
                from agents.vdqn.cfg.gpu_configs import get_gpu_config
                gpu_cfg = get_gpu_config(None)
                gpu_profile = {
                    'gpu_name': gpu_cfg.name,
                    'vram_gb': gpu_cfg.vram_gb,
                    'batch_size': gpu_cfg.batch_size,
                    'memory_size': gpu_cfg.memory_size,
                    'hidden_sizes': list(gpu_cfg.hidden_sizes),
                    'train_repeats': gpu_cfg.train_repeats,
                    'use_amp': gpu_cfg.use_amp,
                    'use_gpu_storage': gpu_cfg.use_gpu_storage,
                    'learning_rate': gpu_cfg.learning_rate,
                }
            except Exception:
                gpu_profile = None

            manifest_path = self.config.save_manifest(
                symbol='multi',
                run_id=self.run_id,
                model_type='multiframe',
                timeframes=self.config.timeframes,
                config_dict={**self.config.to_dict(), **({'gpu_profile': gpu_profile} if gpu_profile else {})},
                symbols=self.config.symbols
            )
            results_payload = {
                'best_val_accuracy': float(best_val_accuracy),
                'epochs_trained': int(len(self.train_losses)),
                'train_loss_last': float(self.train_losses[-1]) if self.train_losses else None,
                'val_loss_last': float(self.val_losses[-1]) if self.val_losses else None,
                'gpu_profile': gpu_profile,
                'training_params': {
                    'batch_size': getattr(self.config, 'batch_size', None),
                    'learning_rate': getattr(self.config, 'learning_rate', None),
                    'hidden_sizes': getattr(self.config, 'hidden_channels', None),
                    'sequence_length': getattr(self.config, 'sequence_length', None),
                    'timeframes': self.config.timeframes,
                }
            }
            results_path = self.config.save_results('multi', self.run_id, results_payload, filename='result_multiframe.json')
            self.logger.info(f"📝 Manifest: {manifest_path}")
            self.logger.info(f"📝 Results:  {results_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Не удалось сохранить manifest/results: {e}")
        
        return {
            'model': self.model,
            'best_val_accuracy': best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
