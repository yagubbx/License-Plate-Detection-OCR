import re

PATTERNS = {
    'az': re.compile(r'^[0-9]{2}[A-Z]{2}[0-9]{3}$'),
    'br': re.compile(r'^[A-Z]{3}[0-9]{4}$'),
}
TO_DIGIT = str.maketrans({'O':'0', 'I':'1', 'Q':'0', 'S':'5', 'B':'8', 'Z':'2'})
AZ_DIGIT = str.maketrans({'O':'0','Q':'0','I':'1','J':'1','S':'5','B':'8','Z':'2','L':'4','A':'4'})
TO_LETTER = str.maketrans({'0':'O', '1':'I', '5':'S', '8':'B', '2':'Z','7':'Z'})

def normalize(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def clean_text(text, plate_format='az'):
    """Only correct known character positions in a seven-character plate."""
    if plate_format not in ('az', 'br', 'generic'):
        raise ValueError('plate_format must be az, br or generic')
    # OCR may also read a small dealer/state label. Keep a unique seven-character
    # text region; preserve ambiguity if several regions could be plate numbers.
    regions=[normalize(line) for line in text.splitlines() if normalize(line) not in ('AZ',)]
    candidates=[line for line in regions if len(line)==7 and any(c.isdigit() for c in line)]
    text = candidates[0] if len(candidates)==1 and plate_format!='generic' else ''.join(regions)
    if plate_format=='az' and text.startswith('AZ') and len(text)==9:
        text=text[2:]
    # A plate frame sometimes becomes a leading/trailing glyph. Remove it only
    # when exactly one unambiguous, already well-formed seven-character run exists.
    if plate_format=='az' and len(text) in (8,9):
        runs={text[i:i+7] for i in range(len(text)-6) if PATTERNS['az'].fullmatch(text[i:i+7])}
        if len(runs)==1: text=runs.pop()
    if len(text) == 7 and plate_format != 'generic':
        letters = {2, 3} if plate_format == 'az' else {0, 1, 2}
        digits=AZ_DIGIT if plate_format=='az' else TO_DIGIT
        text = ''.join(c.translate(TO_LETTER if i in letters else digits)
                       for i, c in enumerate(text))
    return text

def valid_plate(text, plate_format='az'):
    pattern = PATTERNS.get(plate_format)
    return bool(pattern.fullmatch(text)) if pattern else None


def display_plate(text, plate_format='az'):
    return f'{text[:2]}-{text[2:4]}-{text[4:]}' if plate_format=='az' and valid_plate(text,'az') else text
