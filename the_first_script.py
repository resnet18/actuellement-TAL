#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io

# ---------- 修复终端显示（解决输出?的问题）----------
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ========== 可自定义配置 ==========
WHITELIST = [
    "的确", "目的", "标的", "似的", "的话", "的说",
]

SPECIAL_WORDS = {
    "好的": "好の",
    "对的": "对の",
}

STOP_CHARS = "，。！？、；：,.\!?;: \t\n\r吧吗呢啊呀啦哈"
# =================================

def build_placeholder(word, idx):
    return f"__PROTECT_{idx:05d}__"

def protect_whitelist(text):
    """白名单保护：只有词确实存在时才替换为占位符"""
    placeholder_map = {}
    sorted_words = sorted(WHITELIST, key=len, reverse=True)
    for idx, word in enumerate(sorted_words):
        if word in text:  # ⭐ 关键修复
            placeholder = build_placeholder(word, idx)
            text = text.replace(word, placeholder)
            placeholder_map[placeholder] = word
    return text, placeholder_map

def handle_special_words(text):
    """处理特殊词（好的/对的），根据后面内容决定是否替换"""
    placeholder_map = {}
    for word, replacement in SPECIAL_WORDS.items():
        matches = []
        pos = 0
        while True:
            pos = text.find(word, pos)
            if pos == -1:
                break
            end = pos + len(word)
            if end >= len(text) or text[end] in STOP_CHARS:
                action = 'keep'
            else:
                action = 'replace'
            matches.append((pos, end, action))
            pos = end
        # 从后往前替换，避免索引错乱
        for pos, end, action in reversed(matches):
            if action == 'keep':
                placeholder = build_placeholder(word, len(placeholder_map) + 1)
                placeholder_map[placeholder] = word
                text = text[:pos] + placeholder + text[end:]
            else:
                text = text[:pos] + replacement + text[end:]
    return text, placeholder_map

def process_text(text):
    """主处理流水线"""
    # 1. 处理特殊词
    text, special_map = handle_special_words(text)
    # 2. 保护白名单
    text, whitelist_map = protect_whitelist(text)
    # 3. 无脑替换剩下的“的”
    text = text.replace("的", "の")
    # 4. 还原占位符（先还原白名单，再还原特殊词）
    for placeholder, original in whitelist_map.items():
        text = text.replace(placeholder, original)
    for placeholder, original in special_map.items():
        text = text.replace(placeholder, original)
    return text

# ========== 交互 / 文件 / 管道 ==========
def main():
    # 模式1：处理文件
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"错误：文件 '{filename}' 不存在", file=sys.stderr)
            sys.exit(1)
        new_content = process_text(content)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已处理文件：{filename}")
        return

    # 模式2：管道输入（如 echo "xxx" | python 脚本.py）
    if not sys.stdin.isatty():
        content = sys.stdin.read()
        if content:
            sys.stdout.write(process_text(content))
        return

    # 模式3：交互测试（直接输入文字）
    print("=== 的→の 转换器（交互模式）===")
    print("输入要转换的文字，按 Enter 查看结果。")
    print("输入空行或 'quit' 退出。\n")
    while True:
        try:
            line = input("请输入文字 > ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() in ('', 'quit'):
            print("谢谢使用，再见！")
            break
        result = process_text(line)
        print("转换结果：", result)
        print("-" * 40)

if __name__ == "__main__":
    main()