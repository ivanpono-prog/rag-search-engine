import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Query Rewriting CLI"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query to rewrite"
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as file:
        img = file.read()

    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    system_prompt = """
    Given the included image and text query, rewrite the text query
    to improve search results from a movie database.

    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt.strip()
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                },
                {
                    "type": "text",
                    "text": args.query.strip()
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("No text in API response")

    print(f"Rewritten query: {content.strip()}")

    if response.usage is not None:
        print(f"Total tokens: {response.usage.total_tokens}")

if __name__ == "__main__":
    main()
