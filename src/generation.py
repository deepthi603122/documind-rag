from src.config import MIN_SIMILARITY


NOT_FOUND_MESSAGE = "I could not find this information in the provided documents."


class AnswerGenerator:
    """
    Grounded answer generator.

    Uses retrieved document chunks directly, so the application
    does not require an external LLM API or API credits.
    """

    def __init__(self):
        pass

    def _select_relevant_sentences(self, question, results):
        question_words = {
            word.lower().strip(".,?!:;()[]{}")
            for word in question.split()
            if len(word) > 2
        }

        scored_sentences = []

        for result in results:
            text = result.get("text", "")

            sentences = []
            current = ""

            for char in text:
                current += char
                if char in ".!?":
                    sentence = current.strip()
                    if sentence:
                        sentences.append(sentence)
                    current = ""

            if current.strip():
                sentences.append(current.strip())

            for sentence in sentences:
                sentence_words = {
                    word.lower().strip(".,?!:;()[]{}")
                    for word in sentence.split()
                    if len(word) > 2
                }

                overlap = len(question_words.intersection(sentence_words))

                if overlap > 0:
                    scored_sentences.append(
                        (
                            overlap,
                            result.get("score", 0.0),
                            sentence,
                            result,
                        )
                    )

        scored_sentences.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True
        )

        return scored_sentences[:5]

    def generate(self, question, results):
        if not results:
            return NOT_FOUND_MESSAGE

        valid_results = [
            result
            for result in results
            if result.get("score", 0.0) >= MIN_SIMILARITY
        ]

        if not valid_results:
            return NOT_FOUND_MESSAGE

        selected = self._select_relevant_sentences(
            question,
            valid_results
        )

        if not selected:
            return NOT_FOUND_MESSAGE

        answer_sentences = []

        for _, _, sentence, _ in selected:
            if sentence not in answer_sentences:
                answer_sentences.append(sentence)

        answer = " ".join(answer_sentences)

        if not answer.strip():
            return NOT_FOUND_MESSAGE

        return answer.strip()