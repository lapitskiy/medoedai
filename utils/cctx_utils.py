KNOWN_QUOTES = ("USDT", "USD", "USDC", "BUSD", "USDP")

def _strip_separators(s: str) -> str:
    return str(s or "").replace("/", "").replace("-", "").replace("_", "").upper()

def _detect_quote(no_sep: str) -> tuple[str, str]:
    for q in KNOWN_QUOTES:
        if no_sep.endswith(q):
            return no_sep[:-len(q)] or no_sep, q
    return no_sep, ""  # quote not found

def normalize_to_db(symbol: str) -> str:
    """Возвращает форму для БД без разделителей: BASEQUOTE (например, TONUSDT).
    Если квота отсутствует, по умолчанию добавляет USDT.
    """
    no_sep = _strip_separators(symbol)
    base, quote = _detect_quote(no_sep)
    if not quote:
        quote = "USDT"
    return f"{base}{quote}".upper()

def normalize_to_cctx(symbol: str) -> str:
    """Возвращает форму для ccxt: BASE/QUOTE (например, TON/USDT).
    Если квота отсутствует, по умолчанию добавляет USDT.
    """
    db_form = normalize_to_db(symbol)
    base, quote = _detect_quote(db_form)
    quote = quote or "USDT"
    return f"{base}/{quote}".upper()

def normalize_symbol(sym: str) -> str:
    """Совместимость со старым кодом: возвращает ccxt-форму."""
    return normalize_to_cctx(sym)
