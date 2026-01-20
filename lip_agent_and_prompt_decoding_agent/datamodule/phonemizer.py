import re

# Try to use g2p_en; if cmudict not available we fall back to pronouncing package
try:
    from g2p_en import G2p
    _g2p = G2p()
    def _g2p_phonemes(text):
        seq = _g2p(text)
        phones = [re.sub(r"[0-9]$", "", p) for p in seq if re.match(r"^[A-Z]+[0-9]?$", p)]
        return [p.upper() for p in phones]
except Exception:
    _g2p = None
    try:
        import pronouncing
        def _g2p_phonemes(text):
            words = re.findall(r"[A-Za-z']+", text)
            phones = []
            for w in words:
                p_list = pronouncing.phones_for_word(w.lower())
                if p_list:
                    phones.extend([re.sub(r"[0-9]$", "", ph) for ph in p_list[0].split()])
                else:
                    phones.append(w.upper())
            return [p.upper() for p in phones]
    except Exception:
        _g2p_phonemes = None

# pypinyin optional for Chinese
try:
    from pypinyin import lazy_pinyin, Style
    def _zh_phonemes(text):
        pinyins = lazy_pinyin(text, style=Style.NORMAL)
        return [p.lower() for p in pinyins if p.strip()]
except Exception:
    def _zh_phonemes(text):
        return [t for t in re.findall(r"[A-Za-z']+", text)]


def guess_language(text):
    # crude heuristic: if contains CJK characters -> zh, else en
    if re.search(r"[\u4e00-\u9fff]", text):
        return 'zh'
    if re.search(r"[A-Za-z]", text):
        return 'en'
    return 'en'


def phonemize(text, lang=None):
    """Return phoneme sequence string for given text and language."""
    if lang is None:
        lang = guess_language(text)
    if lang.startswith('en'):
        if _g2p_phonemes is None:
            raise RuntimeError("No English phonemizer available (g2p_en or pronouncing required)")
        phones = _g2p_phonemes(text)
        return ' '.join(phones)
    elif lang.startswith('zh'):
        phones = _zh_phonemes(text)
        return ' '.join(phones)
    else:
        if _g2p_phonemes:
            return ' '.join(_g2p_phonemes(text))
        return ' '.join(re.findall(r"[A-Za-z']+", text)).upper()