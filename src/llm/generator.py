"""
LLM sentence generation using GPT-2 and BERT.

AUDIT FIXES (2026-04-24):
- BERT is a Masked Language Model, NOT a text generator (Devlin et al. 2019).
  Its output is relabeled as a 'masked-fill experiment', not natural text generation.
  BERT results should NOT be compared directly to GPT-2 or real corpus text.
- Parser changed from spaCy to Stanza (Qi et al. 2020) for UD-compatible annotations.
  spaCy uses its own dependency scheme mapped to UD via a lossy dictionary, confounding
  parser differences with LLM differences. Stanza outputs native UD annotations.
  spaCy remains as fallback if Stanza is unavailable.
- GPT-2 prompts expanded from 8 to 20 for more representative sampling.
- n_sentences default increased from 200 to 1000 for adequate statistical power.
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
        # AUDIT FIX: increased from 200 to 1000 for adequate statistical power
        self.n_sentences = self.llm_cfg.get("n_sentences", 1000)
        self.max_length = self.llm_cfg.get("max_length", 80)
        self.temperature = self.llm_cfg.get("temperature", 0.9)
        self._gpt2 = None
        self._tokenizer = None
        self._bert = None
        self._bert_tokenizer = None
        self._spacy_nlp = None
        self._stanza_nlp = None

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
        """Generate sentences using GPT-2 autoregressive sampling."""
        self._load_gpt2()
        if self._gpt2 is None:
            return []

        import torch

        if prompts is None:
            # AUDIT FIX: expanded from 8 to 20 prompts for more diverse, representative sampling
            prompts = [
                "The researcher studied", "In the morning, the birds",
                "Scientists believe that", "The government decided",
                "After many years of research,", "The child laughed because",
                "She found the book on", "They were surprised to",
                "The linguist analyzed", "When the committee decided",
                "Most researchers now believe", "The new data suggests",
                "In recent years, scientists have", "The study found that",
                "Languages tend to place", "A careful analysis of",
                "The results confirm", "Children learn to",
                "The government policy will", "Several factors contribute to",
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
        """BERT masked-fill experiment (NOT text generation).

        AUDIT NOTE (2026-04-24): BERT (bert-base-uncased) is a Masked Language Model,
        not an autoregressive text generator (Devlin et al. 2019). This method fills
        [MASK] tokens in hand-crafted templates via BERT's MLM head.
        Results are structurally constrained by template design, NOT by BERT's language
        modeling. Do NOT compare BERT outputs to GPT-2 or real corpus as equivalent
        'LLM generations' — they are fundamentally different.
        Only 75 sentences with 706 interveners produced, vs 555,052 in real corpus.
        This is severely underpowered for distributional comparison.
        Kept for reference only; excluded from primary LLM comparison claims.
        """
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

    def generate_temperature_variants(
        self,
        temperatures: Optional[List[float]] = None,
        n_sentences_each: int = 200,
    ) -> dict:
        """Generate GPT-2 sentences at multiple temperatures for sensitivity analysis.

        AUDIT FIX (2026-04-24) — Fix 10:
        If GPT-2 divergence results change substantially with temperature, conclusions
        are sensitive to this arbitrary hyperparameter. This method generates sentences
        at several temperatures and returns them keyed by temperature value.

        Default temperatures: [0.5, 0.7, 0.9, 1.1]
        (0.9 is the default used in the main comparison.)
        """
        self._load_gpt2()
        if self._gpt2 is None:
            return {}

        if temperatures is None:
            temperatures = [0.5, 0.7, 0.9, 1.1]

        original_temp = self.temperature
        original_n    = self.n_sentences
        self.n_sentences = n_sentences_each

        variants: dict = {}
        for temp in temperatures:
            self.temperature = temp
            logger.info("Generating %d sentences at temperature=%.1f", n_sentences_each, temp)
            sents = self.generate_raw_sentences()
            variants[temp] = sents
            logger.info("  Generated %d sentences at temperature=%.1f", len(sents), temp)

        # Restore originals
        self.temperature  = original_temp
        self.n_sentences  = original_n
        return variants

    def _load_stanza(self):
        """Load Stanza UD-compatible parser (preferred over spaCy for LLM comparison)."""
        if self._stanza_nlp is not None:
            return True
        try:
            import stanza
            # Download English models if not already present
            stanza.download("en", verbose=False)
            self._stanza_nlp = stanza.Pipeline(
                "en",
                processors="tokenize,mwt,pos,lemma,depparse",
                tokenize_pretokenized=False,
                verbose=False,
            )
            logger.info("Stanza UD-compatible parser loaded (preferred for LLM comparison)")
            return True
        except ImportError:
            logger.info("Stanza not installed; falling back to spaCy. "
                        "Install with: pip install stanza")
            return False
        except Exception as e:
            logger.warning("Stanza load failed: %s — falling back to spaCy", e)
            return False

    def parse_sentences(self, raw_sentences: List[str]):
        """Parse raw strings into our Sentence/Token format.

        AUDIT FIX (2026-04-24): Uses Stanza (UD-compatible) in preference to spaCy.
        spaCy uses its own dependency scheme with lossy UD mapping, confounding
        parser differences with LLM differences in cross-corpus comparison.
        Stanza outputs native Universal Dependencies annotations.
        Reference: Qi et al. (2020), De Marneffe et al. (2021).
        """
        from src.data.loader import Sentence, Token

        # Try Stanza first (UD-native)
        use_stanza = self._load_stanza()

        if use_stanza and self._stanza_nlp is not None:
            return self._parse_with_stanza(raw_sentences)
        else:
            # Fallback to spaCy
            self._load_spacy()
            if self._spacy_nlp is None:
                logger.warning("Neither Stanza nor spaCy available. LLM parsing disabled.")
                return []
            logger.warning("Using spaCy fallback — results confounded by parser mismatch")
            return self._parse_with_spacy(raw_sentences)

    def _parse_with_stanza(self, raw_sentences: List[str]):
        """Parse with Stanza for native UD annotations."""
        from src.data.loader import Sentence, Token
        result = []
        for idx, text in enumerate(raw_sentences):
            try:
                doc = self._stanza_nlp(text)
                for sent in doc.sentences:
                    tokens = []
                    for word in sent.words:
                        tokens.append(Token(
                            id=word.id,
                            form=word.text,
                            lemma=word.lemma or word.text,
                            upos=word.upos or "X",
                            xpos=word.xpos or "_",
                            feats=dict(f.split("=") for f in (word.feats or "").split("|") if "=" in f),
                            head=word.head,
                            deprel=word.deprel or "dep",
                            deps="_",
                            misc="_",
                        ))
                    if len(tokens) >= 3:
                        result.append(Sentence(
                            sent_id=f"llm_{idx}",
                            text=sent.text,
                            tokens=tokens,
                        ))
            except Exception as e:
                logger.debug("Stanza parse error: %s", e)
        logger.info("Stanza parsed %d LLM sentences", len(result))
        return result

    def _parse_with_spacy(self, raw_sentences: List[str]):
        """Parse with spaCy as fallback (lossy UD mapping — annotated for transparency)."""
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
                        feats={},  # spaCy doesn't provide UD-compatible morph feats reliably
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
                logger.debug("spaCy parse error: %s", e)
        logger.info("spaCy parsed %d LLM sentences (fallback mode)", len(result))
        return result
