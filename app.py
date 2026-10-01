from src.config import TOP_K
from src.generation import (
    AnswerGenerator,
    NOT_FOUND_MESSAGE,
)
from src.retrieval import Retriever


def print_sources(results):
    """Display retrieved chunks and their citations."""

    if not results:
        return

    print("\n" + "-" * 70)
    print("RETRIEVED SOURCE CHUNKS")
    print("-" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n[{index}] "
            f"{result['source']} "
            f"| Page/Section: {result['page']} "
            f"| Similarity: {result['score']}"
        )

        preview = result["text"].replace(
            "\n",
            " ",
        )

        if len(preview) > 350:
            preview = preview[:350] + "..."

        print(f"    {preview}")

    print("-" * 70)


def main():
    print("=" * 70)
    print("DOCUMIND - BASIC DOCUMENT Q&A BOT")
    print("RAG Pipeline")
    print("=" * 70)

    print("\nLoading vector database...")

    try:
        retriever = Retriever()
        generator = AnswerGenerator()

    except Exception as error:
        print(
            f"\nERROR: {error}"
        )

        print(
            "\nMake sure you have:"
        )

        print(
            "1. Installed requirements"
        )

        print(
            "2. Created the .env file"
        )

        print(
            "3. Added your OpenAI API key"
        )

        print(
            "4. Run: python index.py"
        )

        return

    print("\n[OK] RAG system is ready.")

    print(
        "\nAsk questions about your documents."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    print("=" * 70)

    while True:
        try:
            question = input(
                "\nQuestion: "
            ).strip()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

        except EOFError:
            print("\n\nExiting...")
            break

        if not question:
            print(
                "Please enter a question."
            )
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print(
                "\nThank you for using DocuMind."
            )
            break

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        print(
            "\nSearching documents..."
        )

        try:
            results = retriever.retrieve(
                query=question,
                top_k=TOP_K,
            )

        except Exception as error:
            print(
                f"\nRetrieval error: {error}"
            )
            continue

        if not results:
            print(
                f"\nANSWER:\n{NOT_FOUND_MESSAGE}"
            )

            print(
                "\nNo sufficiently relevant document "
                "chunks were found."
            )

            continue

        # -----------------------------------------------------
        # Generation
        # -----------------------------------------------------

        print(
            "Generating grounded answer..."
        )

        try:
            answer = generator.generate(
                question=question,
                results=results,
            )

        except Exception as error:
            print(
                f"\nGeneration error: {error}"
            )
            continue

        # -----------------------------------------------------
        # Display answer
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(answer)

        # -----------------------------------------------------
        # Display citations/source chunks
        # -----------------------------------------------------

        print_sources(results)


if __name__ == "__main__":
    main()