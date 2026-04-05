import re
import unicodedata

def clean_phones(x: str):
    x = unicodedata.normalize('NFKD', x)
    return re.sub(r"[ˈˌːˑ‿]", "", x)
