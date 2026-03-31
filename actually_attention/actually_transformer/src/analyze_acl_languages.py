import os
import re
import sys
import json
from collections import defaultdict

try:
    import pdfplumber
except ImportError:
    print("请先安装 pdfplumber: pip install pdfplumber")
    sys.exit(1)

# 语言检测范围
LANG_RANGES = {
    '中文': ('\u4e00', '\u9fff'),
    '日文平假名': ('\u3040', '\u309f'),
    '日文片假名': ('\u30a0', '\u30ff'),
    '韩文': ('\uac00', '\ud7af'),
    '阿拉伯文': ('\u0600', '\u06ff'),
    '西里尔文': ('\u0400', '\u04ff'),
    '希腊文': ('\u0370', '\u03ff'),
    '泰文': ('\u0e00', '\u0e7f'),
}

# 法语特殊字符（带重音符号的拉丁字母）
FRENCH_ACCENTS = "àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ"
FRENCH_PATTERNS = re.compile(r'[àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]')

def detect_languages(text):
    """检测文本中包含哪些语言"""
    results = defaultdict(int)
    
    # CJK/阿拉伯等非拉丁语系
    for lang, (start, end) in LANG_RANGES.items():
        chars = re.findall(f'[{start}-{end}]', text)
        if chars:
            results[lang] = len(chars)
    
    # 法语检测（扩展拉丁字符）
    french_chars = FRENCH_PATTERNS.findall(text)
    if french_chars:
        results['法语（重音符号）'] = len(french_chars)
    
    # 德语检测（ß, äöü）
    german_chars = re.findall(r'[ßäöüÄÖÜ]', text)
    if german_chars:
        results['德语（变音符号）'] = len(german_chars)
    
    # 西班牙语检测（ñ, ¿, ¡）
    spanish_chars = re.findall(r'[ñÑ¿¡]', text)
    if spanish_chars:
        results['西班牙语'] = len(spanish_chars)
    
    return dict(results)

def extract_and_analyze(pdf_path):
    """提取并分析 PDF"""
    stats = {
        'total_pages': 0,
        'languages': defaultdict(int),
        'samples': defaultdict(list),
        'pages_with_lang': defaultdict(set)
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            stats['total_pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                
                langs = detect_languages(text)
                for lang, count in langs.items():
                    stats['languages'][lang] += count
                    stats['pages_with_lang'][lang].add(page_num)
                    
                    # 保存示例
                    if len(stats['samples'][lang]) < 2:
                        lines = text.split('\n')
                        for line in lines:
                            pattern = None
                            if lang == '法语（重音符号）':
                                pattern = FRENCH_PATTERNS
                            elif lang == '中文':
                                pattern = re.compile(r'[\u4e00-\u9fff]')
                            elif lang == '日文平假名':
                                pattern = re.compile(r'[\u3040-\u309f]')
                            
                            if pattern and pattern.search(line):
                                stats['samples'][lang].append((page_num, line.strip()[:120]))
                                break
    except Exception as e:
        print(f"  错误: {e}")
    
    return stats

def analyze_acl_diversity(folder_path):
    """分析 ACL 会议的语言多样性"""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    print(f"🔍 扫描 ACL 2025 语言分布 ({len(pdf_files)} 个文件)...")
    print("=" * 70)
    
    all_stats = {}
    conference_totals = defaultdict(int)
    
    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n📄 {pdf_file}")
        
        stats = extract_and_analyze(pdf_path)
        all_stats[pdf_file] = stats
        
        if stats['languages']:
            for lang, count in sorted(stats['languages'].items(), key=lambda x: -x[1]):
                pages = len(stats['pages_with_lang'][lang])
                print(f"  • {lang}: {count} 字符 (出现在 {pages} 页)")
        else:
            print("  • 仅标准英文（ASCII）")
    
    # 总结报告
    print("\n" + "=" * 70)
    print("📊 ACL 2025 多语言统计报告")
    print("=" * 70)
    
    if conference_totals:
        total_non_english = sum(conference_totals.values())
        print(f"非英语字符总数: {total_non_english:,}\n")
        
        # 分类展示
        asian_langs = ['中文', '日文平假名', '日文片假名', '韩文']
        european_langs = ['法语（重音符号）', '德语（变音符号）', '西班牙语', '希腊文', '西里尔文']
        middle_east = ['阿拉伯文', '泰文']
        
        categories = [
            ('🇨🇳 亚洲语言 (CJK)', asian_langs),
            ('🇫🇷 欧洲语言 (拉丁/希腊/西里尔)', european_langs),
            ('🌍 中东/南亚', middle_east)
        ]
        
        for cat_name, lang_list in categories:
            cat_total = sum(conference_totals.get(l, 0) for l in lang_list)
            if cat_total > 0:
                print(f"\n{cat_name}: {cat_total:,} 字符")
                for lang in lang_list:
                    if lang in conference_totals:
                        count = conference_totals[lang]
                        pct = (count / total_non_english) * 100
                        print(f"  • {lang:20s}: {count:6,} ({pct:5.1f}%)")
        
        # 特殊分析：法语友好度
        if '法语（重音符号）' in conference_totals:
            fr_count = conference_totals['法语（重音符号）']
            print(f"\n🥐 法语分析:")
            print(f"  检测到 {fr_count} 个带重音符号的字符")
            print(f"  可能场景：作者名（François, Joséphine）、引用的法语论文、多语言实验")
            
            # 显示法语示例
            print(f"\n  法语出现示例:")
            for pdf_name, stats in all_stats.items():
                if '法语（重音符号）' in stats['samples']:
                    for page, sample in stats['samples']['法语（重音符号）'][:2]:
                        # 高亮法语字符
                        highlighted = FRENCH_PATTERNS.sub(lambda m: f"[{m.group()}]", sample)
                        print(f"    • {pdf_name} p.{page}: {highlighted}")
        
        # 中文友好度对比
        if '中文' in conference_totals or '法语（重音符号）' in conference_totals:
            zh_count = conference_totals.get('中文', 0)
            fr_count = conference_totals.get('法语（重音符号）', 0)
            jp_count = conference_totals.get('日文平假名', 0) + conference_totals.get('日文片假名', 0)
            
            print(f"\n📈 主要非英语语言对比:")
            print(f"  中文汉字:  {zh_count:6,}")
            print(f"  法语重音:  {fr_count:6,}")
            print(f"  日文假名:  {jp_count:6,}")
            
            if max(zh_count, fr_count, jp_count) == zh_count:
                print("  🏆 ACL 2025 最关注: 中文")
            elif max(zh_count, fr_count, jp_count) == fr_count:
                print("  🏆 ACL 2025 最关注: 法语/欧洲语言")
            else:
                print("  🏆 ACL 2025 最关注: 日文")
    
    else:
        print("未检测到非英语字符（不太可能，请检查 PDF 路径）")
    
    # 保存报告
    report_path = os.path.join(folder_path, 'language_analysis.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        export_stats = {}
        for pdf, data in all_stats.items():
            export_stats[pdf] = {
                'languages': dict(data['languages']),
                'pages_with_lang': {k: list(v) for k, v in data['pages_with_lang'].items()},
                'total_pages': data['total_pages']
            }
        json.dump(export_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细报告: {report_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.normpath(os.path.join(script_dir, "../data"))
    
    folder = sys.argv[1] if len(sys.argv) > 1 else default_path
    analyze_acl_diversity(folder)
