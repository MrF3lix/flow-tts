from test_tts.text.cleaners import swiss_german
from test_tts.text.tokenizer import CharTokenizer


def test_cleaner_numbers_and_typography():
    assert swiss_german("Am 3. April: 50'000 Fr. (26,9 %)") == "am dritten april: fünfzigtausend fr. sechsundzwanzig komma neun prozent"
    assert swiss_german("«Grüezi» – sagte sie’s.") == '"grüezi" - sagte sie\'s.'
    assert swiss_german("Strasse und Grösse ß") == "strasse und grösse ss"


def test_tokenizer_blanks_unknown_and_state():
    tok = CharTokenizer.build(["abc", "Ä b"], add_blank=True)
    ids = tok.encode("ab")
    assert ids == [tok.blank_id, tok.index["a"], tok.blank_id, tok.index["b"], tok.blank_id]
    assert tok.encode("x")[1] == tok.unk_id
    assert tok.decode(tok.encode("ä b")) == "ä b"
    clone = CharTokenizer.from_state(tok.state_dict())
    assert clone.vocab == tok.vocab and clone.encode("cab") == tok.encode("cab")
