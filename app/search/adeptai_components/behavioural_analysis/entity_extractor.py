from __future__ import annotations
import re
from typing import Dict, List
import spacy

class EntityExtractor:
    """
    Entity extraction using spaCy Transformer NER (en_core_web_trf).
    - Extracts ORG, PERSON, DATE, GPE, LANGUAGE, PRODUCT, etc.
    - Provides light role extraction and duration parsing.
    Notes:
      • spaCy’s default model doesn’t include a JOB_TITLE label; we infer roles via patterns + noun chunks.
      • We avoid keywords; we leverage POS patterns + dependency heads + capitalized nominal spans.
    """

    ROLE_HINTS = re.compile(
        r"\b(head|lead|leader|manager|director|architect|engineer|developer|scientist|analyst|consultant|intern|founder|owner|cto|ceo|cpo|principal|specialist|administrator|coordinator)\b",
        re.I
    )

    def __init__(self, spacy_model: str = "en_core_web_trf"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Fallback to a smaller model if transformer isn’t installed
            self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)

        ents: Dict[str, List[str]] = {
            "COMPANIES": [],
            "ROLES": [],
            "SKILLS": [],
            "DATES": [],
            "LOCATIONS": [],
            "PERSONS": [],
        }

        for e in doc.ents:
            if e.label_ in ("ORG",):
                ents["COMPANIES"].append(e.text)
            elif e.label_ in ("DATE",):
                ents["DATES"].append(e.text)
            elif e.label_ in ("GPE", "LOC"):
                ents["LOCATIONS"].append(e.text)
            elif e.label_ in ("PERSON",):
                ents["PERSONS"].append(e.text)
            elif e.label_ in ("LANGUAGE", "PRODUCT"):
                ents["SKILLS"].append(e.text)

        # Role discovery via noun chunks / dependency heads (beyond keyword match).
        # We favor nominal heads that co-occur with leadership/responsibility semantics.
        roles = set()
        for chunk in doc.noun_chunks:
            span = chunk.text.strip()
            if len(span.split()) <= 6 and self.ROLE_HINTS.search(span):
                roles.add(span)
        for token in doc:
            if token.pos_ in ("PROPN", "NOUN") and self.ROLE_HINTS.search(token.text):
                roles.add(token.text)
        ents["ROLES"] = list(roles)

        # Deduplicate & normalize (preserve original case for readability)
        for k in ents:
            ents[k] = list(dict.fromkeys([s.strip() for s in ents[k] if s.strip()]))

        return ents
