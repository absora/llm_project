#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: winku
@Date: 2025/3/21 22:42
@Description: 大模型模块
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass


class ZhipuChat(BaseModel):
    def __init__(self, path: str = '', model: str = '') -> None:
        super().__init__(path)
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(
            api_key=os.getenv("ZHIPUAI_API_KEY")
        )
        self.model = "glm-4"

    def chat(self, prompt: str, history: List[dict]):
        history.append(
            {
                'role': 'user',
                'content': prompt
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            temperature=0.1
        )

        reply = response.choices[0].message.content
        history.append(
            {
                'role': 'assistant',
                'content': reply
            }
        )
        return reply, history


# if __name__ == '__main__':
#     model = ZhipuChat()
#     print(model.chat('Hello', []))
