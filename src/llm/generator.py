"""
LLM sentence generation using GPT-2 and BERT (masked-LM fill).
Generated sentences are parsed with spacy's dependency parser,
then converted to our internal Sentence / Token format.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

_UPOS_MAP = {
    "NOUN": "NOUN", "VERB": "VERB", "ADJ": "ADJ", "ADV": "ADV",
    "PRON": "PRON", "DET": "DET", "ADP": "ADP", "CONJ": "CCONJ",
    "CCONJ": "CCONJ", "SCONJ": "SCONJ", "PART": "PART", "NUM": "NUM",
    "PUNCT": "PUNCT", "SYM": "SYM", "INTJ": "INTJ", "PROPN": "PROPN",
    "AUX": "AUX", "X": "X", "SPACE": "PUNCT",
}

_SPACY_DEP_TO_UD = {
    "nsubj": "nsubj", "obj": "obj", "iobj": "iobj", "csubj": "csubj",
    "ccomp": "ccomp", "xcomp": "xcomp", "obl": "obl", "vocative": "vocative",
    "expl": "expl", "dislocated": "dislocated", "advcl": "advcl",
    "advmod": "advmod", "discourse": "discourse", "aux": "aux",
    "cop": "cop", "mark": "mark", "nmod": "nmod", "appos": "appos",
    "nummod": "nummod", "acl": "acl", "amod": "amod", "det": "det",
    "clf": "clf", "case": "case", "conj": "conj", "cc": "cc",
    "fixed": "fixed", "flat": "flat", "compound": "compound",
    "list": "list", "parataxis": "parataxis", "orphan": "orphan",
    "goeswith": "goeswith", "reparandum": "reparandum", "punct": "punct",
    "root": "root", "dep": "dep",
}

# templates for BERT masked-LM sentence generation
BERT_TEMPLATES = [
    "The [MASK] quickly ran across the [MASK] street.",
    "A [MASK] scientist discovered a [MASK] element.",
    "The [MASK] student read the [MASK] book carefully.",
    "Many [MASK] researchers studied the [MASK] problem.",
    "The [MASK] government announced a [MASK] policy.",
    "She [MASK] the report and submitted it [MASK].",
    "The [MASK] child played with a [MASK] toy.",
    "They [MASK] decided to build a [MASK] bridge.",
    "A [MASK] teacher explained the [MASK] concept clearly.",
    "The [MASK] committee reviewed the [MASK] proposal.",
    "Several [MASK] birds flew over the [MASK] mountain.",
    "He [MASK] wrote a letter to his [MASK] friend.",
    "The [MASK] doctor examined the [MASK] patient thoroughly.",
    "Our [MASK] team completed the [MASK] project successfully.",
    "The [MASK] artist painted a [MASK] landscape beautifully.",
]


class LLMGenerator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.llm_cfg = cfg.get("llm", {})
        self.model_name = self.llm_cfg.get("model", "gpt2")
        self.n_sentences = self.llm_cfg.get("n_sentences", 200)
        self.max_length = self.llm_cfg.get("max_length", 80)
        self.temperature = self.llm_cfg.get("temperature", 0.9)
        self._gpt2 = None
        self._tokenizer = None
        self._bert = None
        self._bert_tokenizer = None
        self._spacy_nlp = None

    def _load_gpt2(self):
        if self._gpt2 is not None:
            return
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            logger.info("Loading %s ...", self.model_name)
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self._gpt2 = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._gpt2.eval()
            logger.info("GPT-2 loaded")
        except ImportError:
            logger.warning("transformers not installed; LLM generation disabled")
        except Exception as e:
            logger.warning("GPT-2 load failed: %s", e)

    def _load_bert(self):
        if self._bert is not None:
            return
        try:
            from transformers import BertForMaskedLM, BertTokenizer
            logger.info("Loading BERT (bert-base-uncased) ...")
            self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self._bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
            self._bert.eval()
            logger.info("BERT loaded")
        except ImportError:
            logger.warning("transformers not installed; BERT generation disabled")
        except Exception as e:
            logger.warning("BERT load failed: %s", e)

    def _load_spacy(self):
        if self._spacy_nlp is not None:
            return
        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        except Exception:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_md")
                logger.info("spaCy en_core_web_md loaded")
            except Exception as e:
                logger.warning("spaCy not available: %s — LLM parsing disabled", e)

    def generate_raw_sentences(
        self,
        prompts: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate sentences using GPT-2."""
        self._load_gpt2()
        if self._gpt2 is None:
            return []

        import torch

        if prompts is None:
            prompts = [
                "The researcher studied", "In the morning, the birds",
                "Scientists believe that", "The government decided",
                "After many years of research,", "The child laughed because",
                "She found the book on", "They were surprised to",
            ]

        sentences = []
        per_prompt = max(1, self.n_sentences // len(prompts))
        seed = self.cfg.get("project", {}).get("random_seed", 42)

        for prompt in prompts:
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            for i in range(per_prompt):
                try:
                    torch.manual_seed(seed + i)
                    output = self._gpt2.generate(
                        inputs,
                        max_length=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self._tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                    )
                    text = self._tokenizer.decode(output[0], skip_special_tokens=True)
                    # split on sentence boundaries
                    parts = re.split(r"(?<=[.!?])\s+", text)
                    for part in parts:
                        part = part.strip()
                        if 5 <= len(part.split()) <= 40:
                            sentences.append(part)
                            if len(sentences) >= self.n_sentences:
                                break
                except Exception as e:
                    logger.debug("Generation error: %s", e)
            if len(sentences) >= self.n_sentences:
                break

        return sentences[: self.n_sentences]

    def generate_bert_sentences(
        self,
        templates: Optional[List[str]] = None,
        n_variations: int = 5,
    ) -> List[str]:
        """Generate sentences using BERT masked-LM fill-in."""
        self._load_bert()
        if self._bert is None:
            return []

        import torch

        if templates is None:
            templates = BERT_TEMPLATES

        sentences = []
        seed = self.cfg.get("project", {}).get("random_seed", 42)

        for tmpl_idx, template in enumerate(templates):
            for var in range(n_variations):
                try:
                    torch.manual_seed(seed + tmpl_idx * 100 + var)
                    inputs = self._bert_tokenizer(template, return_tensors="pt")
                    mask_positions = (inputs["input_ids"] == self._bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

                    with torch.no_grad():
                        outputs = self._bert(**inputs)
                    logits = outputs.logits

                    # fill each [MASK] with a sampled token from top-k
                    result_ids = inputs["input_ids"].clone()
                    for mask_pos in mask_positions:
                        mask_logits = logits[0, mask_pos, :]
                        # temperature sampling from top-20
                        top_k = 20
                        top_vals, top_idxs = torch.topk(mask_logits, top_k)
                        probs = torch.softmax(top_vals / self.temperature, dim=0)
                        chosen = top_idxs[torch.multinomial(probs, 1).item()]
                        result_ids[0, mask_pos] = chosen

                    text = self._bert_tokenizer.decode(result_ids[0], skip_special_tokens=True)
                    text = text.strip()
                    if 4 <= len(text.split()) <= 40:
                        sentences.append(text)

                except Exception as e:
                    logger.debug("BERT generation error: %s", e)

                if len(sentences) >= self.n_sentences:
                    break
            if len(sentences) >= self.n_sentences:
                break

        logger.info("BERT generated %d sentences", len(sentences))
        return sentences[:self.n_sentences]

    def parse_sentences(self, raw_sentences: List[str]):
        """Parse raw strings into our Sentence/Token format using spaCy."""
        self._load_spacy()
        if self._spacy_nlp is None:
            return []

        from src.data.loader import Sentence, Token

        result = []
        for idx, text in enumerate(raw_sentences):
            try:
                doc = self._spacy_nlp(text)
                tokens = []
                for tok in doc:
                    if tok.is_space:
                        continue
                    head_id = tok.head.i + 1 if tok.head.i != tok.i else 0
                    upos = _UPOS_MAP.get(tok.pos_, "X")
                    deprel = _SPACY_DEP_TO_UD.get(tok.dep_, "dep")
                    tokens.append(Token(
                        id=tok.i + 1,
                        form=tok.text,
                        lemma=tok.lemma_,
                        upos=upos,
                        xpos=tok.tag_,
                        feats={},
                        head=head_id,
                        deprel=deprel,
                        deps="_",
                        misc="_",
                    ))
                if len(tokens) >= 3:
                    result.append(Sentence(
                        sent_id=f"llm_{idx}",
                        text=text,
                        tokens=tokens,
                    ))
            except Exception as e:
                logger.debug("Parse error: %s", e)

        logger.info("Parsed %d LLM sentences", len(result))
        return result
