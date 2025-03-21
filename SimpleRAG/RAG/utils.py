#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: winku
@Date: 2025/3/21 00:36
@Description: 文档加载和切分
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re

enc = tiktoken.get_encoding("cl100k_base")

class ReadFile:
    """
    class for read file
    """
    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        file_list = []
        # os.walk 递归遍历制定文件夹
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content
            )
            docs.extend(chunk_content)
        return docs

    @classmethod
    def read_file_content(cls, file_path: str):
        if file_path.endswith(".pdf"):
            return cls.read_pdf(file_path)
        elif file_path.endswith(".md"):
            return cls.read_markdown(file_path)
        elif file_path.endswith(".txt"):
            return cls.read_text(file_path)
        else:
            return ValueError("Unsupported file type")

    @classmethod
    def get_chunk(self, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()

        for line in lines:
            line = line.replace(' ', '')
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    # 避免跨单词分割
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
                # 处理最后一个块
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                chunk_text.append(curr_chunk)

            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_pdf(cls, file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text)
            return text


    @classmethod
    def read_text(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class Documents:
    """
    获取已经分类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content