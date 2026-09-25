import re

PATTERNS = {
    'az': re.compile(r'^[0-9]{2}[A-Z]{2}[0-9]{3}$'),
    'br': re.compile(r'^[A-Z]{3}[0-9]{4}$'),
}
TO_DIGIT = str.maketrans({'O':'0', 'I':'1', 'Q':'0', 'S':'5', 'B':'8', 'Z':'2'})
TO_LETTER = str.maketrans({'0':'O', '1':'I', '5':'S', '8':'B', '2':'Z'})

def normalize(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def clean_text(text, plate_format='az'):
    """Only correct known character positions in a seven-character plate."""
    if plate_format not in ('az', 'br', 'generic'):
        raise ValueError('plate_format must be az, br or generic')
    # OCR may also read a small dealer/state label. Keep a unique seven-character
    # text region; preserve ambiguity if several regions could be plate numbers.
    regions=[normalize(line) for line in text.splitlines()]
    candidates=[line for line in regions if len(line)==7 and any(c.isdigit() for c in line)]
    text = candidates[0] if len(candidates)==1 and plate_format!='generic' else normalize(text)
    if len(text) == 7 and plate_format != 'generic':
        letters = {2, 3} if plate_format == 'az' else {0, 1, 2}
        text = ''.join(c.translate(TO_LETTER if i in letters else TO_DIGIT)
                       for i, c in enumerate(text))
    return text

def valid_plate(text, plate_format='az'):
    pattern = PATTERNS.get(plate_format)
    return bool(pattern.fullmatch(text)) if pattern else None
