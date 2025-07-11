def normalize_symbol(sym: str) -> str:
    if '/' not in sym and len(sym) >= 6:
        return sym[:-4] + '/' + sym[-4:]
    return sym
