import torch
import numpy as np
import pandas as pd
import whisper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from typing import List, Union, Optional
import copy
from sklearn.utils import class_weight
import re
from pypinyin import pinyin, Style
import sys
import os
from scipy.special import softmax
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
sys.path.append('charsiu/src/')
from Charsiu import charsiu_forced_aligner
import soundfile as sf          # быстрее pydub для чтения/записи WAV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
HF_TOKEN=''
login(HF_TOKEN)

class _AudioMelDataset(Dataset):
    """
    Датасет, который делает всю CPU-тяжёлую предобработку
    (load_audio -> pad_or_trim -> log_mel_spectrogram) в воркерах DataLoader,
    параллельно на нескольких процессах, вместо одного потока в основном
    цикле. GPU при этом не трогается — тензоры остаются на CPU и
    передаются в основной процесс уже готовыми к батчингу.
    """
    def __init__(self, audio_paths: List[str]):
        self.audio_paths = audio_paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)  # CPU tensor, fixed shape (80, 3000)
        return mel


class WhisperClassifierHead(nn.Module):
    """
    Голова классификатора – несколько линейных слоёв с активацией ReLU и Dropout.

    Аргументы:
        input_dim (int): размерность эмбеддинга (зависит от модели Whisper)
        num_classes (int): число классов
        hidden_dims (list): размеры скрытых слоёв (по умолчанию [256, 128])
        dropout (float): вероятность дропаута (0.5)
    """
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: List[int] = [256, 128], dropout: float = 0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class WhisperTorchClassifier(nn.Module):
    """
    Полный классификатор на основе Whisper + обучаемая голова.

    Аргументы:
        whisper_model_name (str): название модели Whisper ('tiny','base','small','medium','large')
        num_classes (int): число классов (определяется автоматически при обучении)
        hidden_dims (list): размеры скрытых слоёв головы
        dropout (float): вероятность дропаута
        device (str): 'cuda', 'cuda:0', 'cpu', и т.п. — если явно указан
            конкретный индекс GPU (например 'cuda:1'), используется только он,
            и авто-мультиGPU не включается, если device_ids не задан явно.
        device_ids (list[int] | None): список индексов GPU для параллельного
            инференса Whisper-энкодера через nn.DataParallel (например [0, 1]).
            Если None и device тоже не указывает конкретный индекс, а видно
            несколько GPU — используются они все автоматически. Голова
            (классификатор) при этом обучается на одном (первом) GPU — она
            маленькая, и мульти-GPU для неё не нужен.
        lr (float): скорость обучения для головы
        epochs (int): число эпох обучения (по умолчанию 50)
        batch_size (int): размер батча при обучении
        num_workers (int): число процессов для параллельной загрузки/mel-препроцессинга аудио
        embed_batch_size (int): размер батча для извлечения эмбеддингов (можно больше, чем batch_size обучения)
    """
    def __init__(self,
                 whisper_model_name: str = 'medium',
                 num_classes: Optional[int] = None,
                 hidden_dims: List[int] = [256, 256],
                 dropout: float = 0.5,
                 device: Optional[str] = None,
                 device_ids: Optional[List[int]] = None,
                 lr: float = 4e-6,
                 epochs: int = 2000,
                 batch_size: int = 32,
                 num_workers: Optional[int] = None,
                 embed_batch_size: int = 32):
        super().__init__()
        # --- Определение устройств (поддержка нескольких GPU) ---
        # Если пользователь явно закрепил конкретный индекс GPU (например
        # device='cuda:1'), уважаем это и НЕ включаем авто-мульти-GPU, если
        # device_ids не задан отдельно. Иначе, если видно несколько GPU,
        # используем их все по умолчанию.
        pinned_to_single_gpu = bool(device) and device.startswith('cuda:')

        if device_ids is not None:
            self.device_ids = list(device_ids)
        elif not pinned_to_single_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = []

        if self.device_ids:
            self.device = f'cuda:{self.device_ids[0]}'
            print(f"Используем {len(self.device_ids)} GPU для инференса энкодера Whisper: {self.device_ids}")
        else:
            self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Загружаем и замораживаем Whisper
        self.whisper = whisper.load_model(whisper_model_name)
        self.whisper.to(self.device)
        for param in self.whisper.parameters():
            param.requires_grad = False
        self.whisper.eval()

        # Определяем размерность эмбеддинга по модели
        # У моделей whisper размер эмбеддинга: tiny->384, base->512, small->768, medium->1024, large->1280
        self.input_dim = self.whisper.dims.n_audio_state

        # Энкодер, используемый для извлечения эмбеддингов. Если доступно
        # несколько GPU, оборачиваем его в nn.DataParallel — тогда каждый
        # батч мел-спектрограмм автоматически разбивается по устройствам,
        # прогоняется параллельно, и результаты собираются обратно на
        # self.device. Голова классификатора (маленькая, per-embedding)
        # намеренно НЕ оборачивается — обучается на одном устройстве.
        if len(self.device_ids) > 1:
            self.encoder = nn.DataParallel(self.whisper.encoder, device_ids=self.device_ids)
        else:
            self.encoder = self.whisper.encoder

        # Голова создаётся после того, как узнаем num_classes (при вызове fit)
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.embed_batch_size = embed_batch_size
        # По умолчанию используем все ядра, кроме одного, для препроцессинга аудио
        self.num_workers = num_workers if num_workers is not None else max(1, (os.cpu_count() or 2) - 1)

        self.head = None  # будет создан в fit
        self.optimizer = None
        self.is_fitted = False

    def _extract_embeddings(self, audio_paths: List[str]) -> np.ndarray:
        """
        Извлекает эмбеддинги для списка файлов.

        Ключевое отличие от предыдущей версии: препроцессинг аудио (CPU-bound:
        load_audio + pad_or_trim + log_mel_spectrogram) вынесен в DataLoader
        с несколькими воркерами (num_workers), которые работают параллельно
        в отдельных процессах. Основной процесс тем временем гоняет encoder
        Whisper на GPU целыми батчами, а не по одному файлу за раз. Это убирает
        сериализацию "CPU готовит один файл -> GPU считает один файл -> повтор"
        и позволяет GPU реально простаивать меньше, а CPU работать параллельно
        на нескольких ядрах.
        """
        dataset = _AudioMelDataset(audio_paths)
        loader = DataLoader(
            dataset,
            batch_size=self.embed_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
        )

        embeddings = []
        with torch.no_grad():
            for mel_batch in tqdm(loader, total=len(loader)):
                mel_batch = mel_batch.to(self.device, non_blocking=True)  # (B, 80, 3000)
                encoder_out = self.encoder(mel_batch)  # (B, seq, dim)
                emb = encoder_out.mean(dim=1).cpu().numpy()  # (B, dim)
                embeddings.append(emb)

        return np.concatenate(embeddings, axis=0).astype(np.float32)

    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """Извлекает эмбеддинг из одного аудиофайла (для единичного инференса/предикта на лету)."""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        with torch.no_grad():
            encoder_out = self.encoder(mel.unsqueeze(0))  # (1, seq, dim)
            emb = encoder_out.mean(dim=1).squeeze().cpu().numpy()  # (dim,)
        return emb

    def fit(self,
            audio_paths: List[str],
            labels: List[Union[str, int]],
            validation_data: Optional[tuple] = None,
            early_stopping: bool = False,
            patience: int = 5,
            restore_best_weights: bool = True) -> dict:
        """
        Обучает голову на предоставленных данных.

        Аргументы:
            audio_paths, labels: обучающие данные (как раньше).
            validation_data (tuple | None): опциональная пара
                (val_audio_paths, val_labels) для контроля переобучения.
                Если передана, эмбеддинги валидации извлекаются один раз до
                начала цикла эпох (не пересчитываются каждую эпоху), а после
                каждой эпохи считаются val_loss и val_accuracy на голове
                в режиме eval().
            early_stopping (bool): если True и передана validation_data,
                обучение останавливается, если val_loss не улучшается
                `patience` эпох подряд (признак начавшегося переобучения).
            patience (int): число эпох без улучшения val_loss до остановки.
            restore_best_weights (bool): если True, по окончании обучения
                (в т.ч. при early stopping) веса головы откатываются к
                состоянию с лучшим val_loss, а не остаются на последней эпохе.

        Возвращает:
            dict с историей обучения: 'train_loss' (List[float]),
            и, если была validation_data, 'val_loss' (List[float]) и
            'val_accuracy' (List[float]) по эпохам. Также сохраняется
            в self.history.
        """
        # Определяем число классов
        unique_labels = sorted(set(labels))
        print(f'num classes: {len(unique_labels)}')
        self.num_classes = len(unique_labels)
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.label_to_idx = label_to_idx
        y=np.array([np.float32(label_to_idx[lbl]) for lbl in labels])
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,  dtype=torch.float32).to(self.device))
        # Создаём голову (пересоздаём, если уже была - например при повторном обучении)
        self.head = WhisperClassifierHead(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.head.parameters(), lr=self.lr)

        # Извлекаем эмбеддинги train (батчами, с параллельным препроцессингом на CPU)
        X_np = self._extract_embeddings(audio_paths)
        X = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        print('embeding complete')
        # Создаём DataLoader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Если передана валидация — извлекаем её эмбеддинги ОДИН раз здесь,
        # а не заново на каждой эпохе (иначе контроль качества сам стал бы
        # узким местом, пересчитывая Whisper-энкодер эпоха за эпохой).
        has_val = validation_data is not None
        if has_val:
            val_paths, val_labels_raw = validation_data
            if len(val_paths) != len(val_labels_raw):
                raise ValueError(
                    f"Количество валидационных файлов ({len(val_paths)}) не "
                    f"совпадает с количеством меток ({len(val_labels_raw)})."
                )
            unknown = sorted(set(val_labels_raw) - set(label_to_idx.keys()), key=str)
            if unknown:
                print(f"Внимание: в валидации есть метки, не виденные при "
                      f"обучении, они будут исключены из расчёта val-метрик: {unknown}")
            keep_idx = [i for i, lbl in enumerate(val_labels_raw) if lbl in label_to_idx]
            if len(keep_idx) == 0:
                print("Внимание: после фильтрации не осталось валидационных "
                      "примеров с известными метками — валидация отключена.")
                has_val = False
            else:
                val_paths_f = [val_paths[i] for i in keep_idx]
                val_labels_f = [val_labels_raw[i] for i in keep_idx]
                Xval_np = self._extract_embeddings(val_paths_f)
                X_val = torch.tensor(Xval_np, dtype=torch.float32).to(self.device)
                y_val = torch.tensor(
                    [label_to_idx[lbl] for lbl in val_labels_f], dtype=torch.long
                ).to(self.device)

        history = {'train_loss': []}
        if has_val:
            history['val_loss'] = []
            history['val_accuracy'] = []

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_no_improve = 0

        # Цикл обучения
        for epoch in range(self.epochs):
            self.head.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                logits = self.head(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(dataloader)
            history['train_loss'].append(train_loss)

            log_line = f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}"

            if has_val:
                # Контроль качества на валидации: считаем loss и accuracy
                # в режиме eval() (без dropout), чтобы честно оценить обобщающую
                # способность головы, а не её поведение во время обучения.
                self.head.eval()
                with torch.no_grad():
                    val_logits = self.head(X_val)
                    val_loss = self.criterion(val_logits, y_val).item()
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_accuracy = (val_preds == y_val).float().mean().item()
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                log_line += f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"

                # Признак переобучения: train_loss продолжает падать, а
                # val_loss перестал улучшаться / растёт.
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    if restore_best_weights:
                        best_state_dict = copy.deepcopy(self.head.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        log_line += (
                            f"  [Val loss не улучшается {patience} эпох подряд — "
                            f"похоже на переобучение]"
                        )

            print(log_line)

            if has_val and early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping на эпохе {epoch+1}: "
                      f"val_loss не улучшался {patience} эпох подряд.")
                break

        if has_val and restore_best_weights and best_state_dict is not None:
            self.head.load_state_dict(best_state_dict)
            print(f"Восстановлены веса лучшей эпохи (val_loss={best_val_loss:.4f}).")

        self.is_fitted = True
        # Сохраняем маппинг индексов обратно в метки
        self.idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
        self.history = history
        return history

    def predict(self, audio_paths: List[str]) -> List[Union[str, int]]:
        """Предсказывает метки для новых аудиофайлов."""
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        self.head.eval()
        X_np = self._extract_embeddings(audio_paths)
        X = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.head(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return [self.idx_to_label[idx] for idx in preds]

    def predict_proba(self, audio_paths: List[str]) -> np.ndarray:
        """Возвращает вероятности классов."""
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        self.head.eval()
        X_np = self._extract_embeddings(audio_paths)
        X = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.head(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход – принимает тензор эмбеддингов (batch, input_dim),
        возвращает логиты. Удобно для использования в составе других моделей.
        """
        if self.head is None:
            raise RuntimeError("Голова не инициализирована. Сначала вызовите fit().")
        return self.head(x)

    def evaluate(self,
                 audio_paths: List[str],
                 labels: List[Union[str, int]],
                 average: str = 'macro',
                 print_report: bool = True) -> dict:
        """
        Контроль качества модели на валидационных данных.

        Прогоняет валидационный набор через predict(), сравнивает с истинными
        метками и считает основные метрики классификации: accuracy, precision,
        recall, f1 (усреднённые по классам способом `average`), а также
        confusion matrix и per-class classification report.

        Аргументы:
            audio_paths (List[str]): пути к валидационным аудиофайлам
            labels (List[Union[str,int]]): истинные метки для этих файлов
                (должны быть из того же множества классов, на котором
                обучалась модель — метки, не встречавшиеся в train, будут
                считаться ошибками предсказания)
            average (str): способ усреднения метрик по классам
                ('macro', 'micro', 'weighted') — передаётся в sklearn
            print_report (bool): печатать ли человекочитаемый отчёт в stdout

        Возвращает:
            dict со следующими полями:
                - accuracy (float)
                - precision (float)
                - recall (float)
                - f1 (float)
                - confusion_matrix (np.ndarray)
                - labels_order (List): порядок классов, использованный в confusion_matrix
                - per_class (dict): precision/recall/f1/support по каждому классу
                - y_true, y_pred (List): исходные и предсказанные метки
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")

        if len(audio_paths) != len(labels):
            raise ValueError(
                f"Количество аудиофайлов ({len(audio_paths)}) не совпадает "
                f"с количеством меток ({len(labels)})."
            )

        y_true = list(labels)
        y_pred = self.predict(audio_paths)

        # Полный список классов: те, что видела модель при обучении, плюс
        # любые неожиданные метки, которые встретились в валидации (чтобы
        # метрики честно наказывали такие случаи, а не падали с ошибкой)
        known_labels = list(self.idx_to_label.values())
        extra_labels = sorted(set(y_true) - set(known_labels), key=str)
        labels_order = known_labels + extra_labels

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_order, average=average, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)

        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(
                y_true, y_pred, labels=labels_order, average=None, zero_division=0
            )
        per_class = {
            lbl: {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i]),
                'support': int(per_class_support[i]),
            }
            for i, lbl in enumerate(labels_order)
        }

        if print_report:
            print(f"Validation accuracy: {accuracy:.4f}")
            print(f"Validation precision ({average}): {precision:.4f}")
            print(f"Validation recall ({average}): {recall:.4f}")
            print(f"Validation f1 ({average}): {f1:.4f}")
            if extra_labels:
                print(f"Внимание: в валидации встретились метки, "
                      f"не виденные при обучении: {extra_labels}")
            print("\nClassification report:")
            print(classification_report(
                y_true, y_pred, labels=labels_order, zero_division=0
            ))
            print("Confusion matrix (rows=true, cols=predicted), order:", labels_order)
            print(cm)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm,
            'labels_order': labels_order,
            'per_class': per_class,
            'y_true': y_true,
            'y_pred': y_pred,
        }
class AudioAnalyzer():
    def __init__(self, model_path, service_folder):
        model_path = hf_hub_download(repo_id="MaximBibikov228/Chin_whisper_phone", filename="model.pth")
        self.service_folder=service_folder
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print(f'use device:{self.device}')
        self.model=torch.load(
            model_path,
            weights_only=False,
            map_location='cpu'
            )
        self.model = self._move_whisper_classifier_to_device(self.model, self.device)
        self.model_parser=charsiu_forced_aligner(
            aligner='charsiu/zh_xlsr_fc_10ms',
            lang='zh',
            device=self.device         # <-- ключевое ускорение
        )
        self.model.eval()
        os.makedirs(service_folder, exist_ok=True)
    def _split_to_initials_finals(self, text: str):
        """
        Разбирает китайские иероглифы на инициали и финали с тоном.
        Некитайские символы игнорируются.

        Возвращает список словарей:
            [{'char': иероглиф, 'initial': инициаль, 'final': финаль_с_тоном}, ...]
        Для слогов без инициали возвращается ''.
        Нейтральный тон обозначается как 0 (вместо внутреннего 5).
        """
        result = []
        for ch in text:
            # Проверяем, является ли символ китайским иероглифом (CJK Unified Ideographs)
            if not re.match(r'[\u4e00-\u9fff]', ch):
                continue   # пропускаем знаки препинания, цифры, буквы и т.д.

            # Инициаль (пустая строка, если слог начинается с гласного)
            initial = pinyin(ch, style=Style.INITIALS, heteronym=False)[0][0]

            # Финали с номером тона (1-4, 5 для нейтрального)
            final_tone = pinyin(ch, style=Style.FINALS_TONE3, heteronym=False)[0][0]

            # Заменяем 5 (нейтральный тон в pypinyin) на 0, как в примере

            if initial!='':
                result.append(initial)
            result.append(final_tone)
            
        return result
    def _move_whisper_classifier_to_device(self, model, device: str = 'cpu'):
        """
        Moves an already-trained (loaded from .pth) WhisperTorchClassifier
        to a new device (e.g. from GPU to CPU) without retraining.

        Handles:
        - unwrapping nn.DataParallel back to a plain module (DataParallel
            requires CUDA, so it can't just be moved to CPU as-is)
        - moving self.whisper (and therefore its encoder) to the new device
        - moving self.head, if it exists (i.e. model was already fit)
        - moving the class-weight tensor inside self.criterion, if it exists
        - updating self.device / self.device_ids bookkeeping so future
            calls (predict, _extract_embeddings, etc.) use the right device
        """
        device = torch.device(device)

        # 1. Unwrap DataParallel if present
        if isinstance(model.encoder, torch.nn.DataParallel):
            model.encoder = model.encoder.module

        # 2. Move the whole whisper model (encoder lives inside it)
        model.whisper.to(device)
        model.encoder = model.whisper.encoder  # re-point to the moved encoder

        # 3. Move the classifier head, if already created (post-fit)
        if getattr(model, 'head', None) is not None:
            model.head.to(device)

        # 4. Move class weights inside the loss criterion, if present
        criterion = getattr(model, 'criterion', None)
        if criterion is not None and getattr(criterion, 'weight', None) is not None:
            criterion.weight = criterion.weight.to(device)

        # 5. Update device metadata used elsewhere in the class
        model.device = str(device)
        model.device_ids = []  # DataParallel/multi-GPU no longer applies

        return model
    def _top_acc(self, y, pred):
        cnt=0
        t5=list(map(lambda x: list(map(lambda y: y[1], x)), pred))
        for i in range(len(y)):
            cnt+= y[i] in t5[i]
        return cnt/len(y)
    def _top1(self, pred):
        return list(map(lambda x: x[0][1], pred))
    def _p_acc(self, y, pred, label_to_idx):
        sm=0
        cnt=0
        for i in range(len(y)):
            if y[i] not in list(label_to_idx.keys()):
                continue
            cnt+=1
            h=label_to_idx[y[i]]
            sm+=pred[i][h]
            #print(pred[i][h])
        return sm/cnt
    def predict(self, text: str, audio_path:str):
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        times = self.model_parser.align(audio=audio, text=text)[0]
        times = [seg for seg in times if seg[2] != '[SIL]']
        phonem=self._split_to_initials_finals(text)
        n=len(times)
        names=[]
        pred=[]
        last=''
        for i in range(n):
            # Сравнение без цифр (тональных помет)
            if times[i][2]==last:
                continue
            last=times[i][2]
            start_sample = int(times[i][0] * sr)
            end_sample   = int(times[i][1] * sr)
            clip = audio[start_sample:end_sample]
            name = f"{self.service_folder.rstrip('/')}/clip_{i}.wav"
            names.append(name)
            sf.write(name, clip, sr)
        pred=self.model.predict_proba(names)
        # k=10
        # pred=list(map(lambda x: sorted([(p, label) for p, label in zip(x, self.model.idx_to_label.values())], reverse=True)[:k], pred)) # top k
        # if len(pred)!=len(phonem):
        #     print("warning: lengths don't match, cropping" )
        #     while len(pred)!=len(phonem):
        #         b_i=sorted([(self._top_acc(phonem, pred[:i]+pred[i+1:]), i) for i in range(len(pred))], reverse=True)[0][1]
        #         pred=pred[:b_i]+pred[b_i+1:]
        # return self._top_acc(phonem, pred)
        al_final=[]
        al_initial=[]
        for idx, label in self.model.idx_to_label.items():
            if label[-1] in '012345':
                al_final.append(idx)
            else:
                al_initial.append(idx)
        pred1=list(pred.copy())
        while len(pred1)!=len(phonem):
            h=sorted([(self._p_acc(phonem, pred1[:i]+pred1[i+1:], self.model.label_to_idx), i) for i in range(len(pred1))], reverse=True)
            b_i=h[0][1]
            #print(h)
            pred1=pred1[:b_i]+pred1[b_i+1:]
        initial_sm=0
        final_sm=0
        initial_cnt=0
        final_cnt=0
        cnt=0
        for i in range(len(phonem)):
            lst_fin=[]
            lst_in=[]
            ch=phonem[i]
            if ch not in list(self.model.label_to_idx.keys()):
                continue
            cnt+=1
            h=self.model.label_to_idx[ch]
            sv_ind=0
            for j in range(len(pred1[i])):
                if j in al_final:
                    lst_fin.append(pred1[i][j])
                    if h==j:
                        sv_ind=len(lst_fin)-1
                else:
                    lst_in.append(pred1[i][j])
                    if h==j:
                        sv_ind=len(lst_in)-1
            lst_in=softmax(np.array(lst_in)*1e2)
            lst_fin=softmax(np.array(lst_fin)*1e2)
            if ch[-1] in '012345':
                final_sm+=lst_fin[sv_ind]
                final_cnt+=1
            else:
                initial_sm+=lst_in[sv_ind]
                initial_cnt+=1
        initial_res=initial_sm/initial_cnt
        final_res=final_sm/final_cnt
        return (initial_res, final_res)