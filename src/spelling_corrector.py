"""Rule-based spelling correction.

A classic noisy-channel corrector (Norvig-style): candidate words are generated
with edit distance 1-2 from each token, then ranked by word frequency. No
machine learning - just rules over letter edits plus a frequency table.
"""

import re
from collections import Counter
from pathlib import Path

# Default vocabulary: the 20,000 most common English words (derived from
# Google's Trillion Word Corpus, via first20hours/google-10000-english),
# ranked by frequency. A small built-in fallback keeps the corrector working
# even if the file is missing; pass a real frequency file (word<space>count
# per line, e.g. Norvig's big.txt) to SpellingCorrector for full coverage.
DEFAULT_WORDLIST = Path(__file__).resolve().parent.parent / "data" / "wordlist.txt"

FALLBACK_WORDS = (
    "the of and to in is you that it he was for on are as with his they I "
    "at be this have from or one had by word but not what all were we when "
    "your can said there use an each which she do how their if will up other "
    "about out many then them these so some her would make like him into time "
    "has look two more write go see number no way could people my than first "
    "water been call who oil its now find long down day did get come made may "
    "part over new sound take only little work know place year live me back "
    "give most very after thing our just name good sentence man think say great "
    "where help through much before line right too mean old any same tell boy "
    "follow came want show also around form three small set put end does "
    "another well large must big even such because turn here why ask went men "
    "read need land different home us move try kind hand picture again change "
    "off play spell air away animal house point page letter mother answer "
    "found study still learn should world high every near add food between own "
    "below country plant last school father keep tree never start city earth "
    "eye light thought head under story saw left few while along might close "
    "something seem next hard open example begin life always those both paper "
    "together got group often run important side children car feet night walk "
    "white sea began grow took river four carry state once book hear stop "
    "without second later miss idea enough eat face watch far real almost let "
    "above girl sometimes mountain cut young talk soon list song leave family "
    "body music color stand sun question fish area mark dog horse birds problem "
    "complete room knew since ever piece told usually didn't friend easy heard "
    "order red door sure become top ship across today during short better best "
    "however low hours black products happened whole measure remember early "
    "waves reached listen wind rock space covered fast several hold himself "
    "toward five step morning passed vowel true hundred against pattern numeral "
    "table north slowly money map busy pulled draw voice seen cold cried plan "
    "notice south sing war ground fall king town I'll unit figure certain field "
    "travel wood fire upon done English road half ten fly gave box finally wait "
    "correct oh quickly person became shown minutes strong verb stars front "
    "feel fact inches street decided contain course surface produce building "
    "ocean class note nothing rest carefully scientists inside wheels stay "
    "green known island week less machine base ago stood plane system behind "
    "ran round boat game force brought understand warm common bring explain "
    "dry though language shape deep thousands yes clear equation yet government "
    "filled heat full hot check object am rule among noun power cannot able six "
    "size dark ball material special heavy fine pair circle include built"
).split()

LETTERS = "abcdefghijklmnopqrstuvwxyz"


class SpellingCorrector:
    def __init__(self, frequency_file=None, wordlist_file=DEFAULT_WORDLIST):
        if frequency_file:
            words = Counter()
            for line in Path(frequency_file).read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    words[parts[0]] += int(parts[1])
            self.word_freq = words
        else:
            path = Path(wordlist_file)
            words = path.read_text(encoding="utf-8").split() if path.exists() else FALLBACK_WORDS
            # Rank order is the frequency signal: earlier word, higher frequency
            self.word_freq = Counter({w: len(words) - i for i, w in enumerate(words)})

    @staticmethod
    def _edits1(word):
        """All strings one edit (delete, transpose, replace, insert) from word."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [l + r[1:] for l, r in splits if r]
        transposes = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
        replaces = [l + c + r[1:] for l, r in splits if r for c in LETTERS]
        inserts = [l + c + r for l, r in splits for c in LETTERS]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}

    def _known(self, words):
        return {w for w in words if w in self.word_freq}

    def correct_word(self, word):
        """Return the most likely intended spelling for a single token."""
        lower = word.lower()
        if lower in self.word_freq:
            return word
        candidates = (
            self._known(self._edits1(lower))
            or self._known(self._edits2(lower))
            or {lower}
        )
        best = max(candidates, key=lambda w: self.word_freq[w])
        # Preserve capitalization
        return best.capitalize() if word[0].isupper() else best

    def correct_text(self, text):
        """Correct every alphabetic token in text, leaving punctuation intact."""
        def fix(match):
            return self.correct_word(match.group(0))

        return re.sub(r"[A-Za-z]+", fix, text)


if __name__ == "__main__":
    corrector = SpellingCorrector()
    print(corrector.correct_text("Ths sentnce has som speling errros."))
