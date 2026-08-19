import re
from deep_translator import GoogleTranslator

# Translation operates ONLY on the segment text (strings). It never receives or
# returns timing metadata, so segment start/end and ordering are preserved by
# construction. Timeline association happens in main.py, which keeps the
# original segment dict (start/end/text) and stores translated_text on it.
MAX_CHUNK_LENGTH = 4500

_translators = {}

def get_translator(target_lang):
    if target_lang not in _translators:
        _translators[target_lang] = GoogleTranslator(source="auto", target=target_lang)
    return _translators[target_lang]


def chunk_text(text, max_len=MAX_CHUNK_LENGTH):
    """Split text into <= max_len chunks at sentence boundaries.

    Splits on punctuation followed by whitespace (". " etc.) so translations
    don't cut off mid-sentence. Returns a list of chunk strings.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_len:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _translate_single(text, target_lang):
    translator = get_translator(target_lang)
    last_error = None

    for attempt in range(1, 3):
        try:
            result = translator.translate(text)
            if not result:
                last_error = f"Translation returned an empty or None result for target_lang '{target_lang}'"
                raise RuntimeError(last_error)
            return result
        except RuntimeError:
            if attempt < 2:
                print(f"[WARN] Translation attempt {attempt}/2 failed: {last_error}. Retrying...")
                continue
            raise
        except Exception as e:
            last_error = f"Translation request failed: {e}"
            print(f"[WARN] Translation attempt {attempt}/2 failed: {e}")
            if attempt < 2:
                continue
            break

    raise RuntimeError(f"Translation failed for target_lang '{target_lang}': {last_error}")


def translate(text, target_lang):
    chunks = chunk_text(text)

    if len(chunks) == 1:
        return _translate_single(chunks[0], target_lang)

    translated_chunks = [_translate_single(chunk, target_lang) for chunk in chunks]
    return " ".join(translated_chunks)
