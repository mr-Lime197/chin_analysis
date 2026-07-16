import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from light_CNN import AudioCNN, InferenceMelSpecDataset, predict
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
# HF_TOKEN=''
# login(HF_TOKEN)
class AudioAnalyzer():
    def __init__(self, model_path='', service_folder='./results_wav'):
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_init=AudioCNN(n_classes=21, class_names=['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh'])
        self.model_init.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model_parser=charsiu_forced_aligner(
            aligner='charsiu/zh_xlsr_fc_10ms',
            lang='zh',
            device=self.device         # <-- ключевое ускорение
        )
        self.model_init.eval()
        self.service_folder=service_folder
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
        init_names=[]
        fin_names=[]
        names=[]
        audio_labels=dict()
        last=''
        for i in range(n):
            if times[i][2]==last:
                continue
            last=times[i][2]
            name = f"{self.service_folder.rstrip('/')}/clip_{i}.wav"
            audio_labels[name]=last
            if last[-1] in '012345':
                fin_names.append(name)
            else:
                init_names.append(name)
            start_sample = int(times[i][0] * sr)
            end_sample   = int(times[i][1] * sr)
            clip = audio[start_sample:end_sample]
            names.append(name)
            sf.write(name, clip, sr)
        #print(init_names, self.model_init.class_names)
        pred_init=predict(self.model_init, init_names, device=self.device, class_names=self.model_init.class_names, num_workers=0)
        sum_init=0
        #print(pred_init)
        for name in init_names:
            sum_init+=pred_init[name][audio_labels[name]]
        return sum_init/len(init_names)