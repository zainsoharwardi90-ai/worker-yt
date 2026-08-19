import edge_tts
import asyncio

VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-SA-ZariyahNeural",
    "hi": "hi-IN-SwaraNeural",
    "ur": "ur-PK-AsadNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "sv": "sv-SE-SofieNeural",
    "da": "da-DK-ChristelNeural",
    "fi": "fi-FI-SelmaNeural",
    "cs": "cs-CZ-VlastaNeural",
    "ro": "ro-RO-AlinaNeural",
    "hu": "hu-HU-NoemiNeural",
    "th": "th-TH-PremwadeeNeural",
    "vi": "vi-VN-HoaiMyNeural",
}

RATE = "+8%"
PITCH = "-15Hz"

def get_voice(target_lang):
    return VOICE_MAP.get(target_lang, "en-US-JennyNeural")

async def _generate(text, output_path, voice, rate=RATE, pitch=PITCH):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)

def generate_speech(text, output_path, target_lang):
    voice = get_voice(target_lang)
    asyncio.run(_generate(text, output_path, voice))
    return output_path
