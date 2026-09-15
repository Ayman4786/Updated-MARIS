from groq import Groq
from dotenv import load_dotenv

import os
import base64
from pathlib import Path


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:

    print("❌ GROQ API KEY NOT FOUND")


# ==================================================
# GROQ CLIENT
# ==================================================

client = Groq(
    api_key=api_key
)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_NAME = "qwen/qwen3.6-27b"

# Maximum number of tokens generated for the final answer
MAX_COMPLETION_TOKENS = 1000


# ==================================================
# REMOVE QWEN THINKING OUTPUT
# ==================================================

def remove_thinking(text: str) -> str:

    if not text:
        return ""

    # --------------------------------------------------
    # Case 1:
    # <think>...</think>ANSWER
    # --------------------------------------------------

    if "<think>" in text:

        if "</think>" in text:

            text = text.split(
                "</think>",
                1
            )[1]

        else:

            text = text.split(
                "<think>",
                1
            )[0]

    # --------------------------------------------------
    # Case 2:
    # Remove accidental whitespace
    # --------------------------------------------------

    return text.strip()


# ==================================================
# ENCODE IMAGE AS BASE64
# ==================================================

def encode_image(image_path: str) -> str:

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_bytes = image_file.read()

    return base64.b64encode(
        image_bytes
    ).decode("utf-8")


# ==================================================
# DETECT IMAGE MIME TYPE
# ==================================================

def get_image_mime_type(
    image_path: str
) -> str:

    extension = Path(
        image_path
    ).suffix.lower()

    if extension == ".png":

        return "image/png"

    if extension in [
        ".jpg",
        ".jpeg"
    ]:

        return "image/jpeg"

    if extension == ".webp":

        return "image/webp"

    # Default
    return "image/png"


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(
    prompt,
    image_paths=None
):

    try:

        # ==================================================
        # REQUEST INFORMATION
        # ==================================================

        print("\n")
        print("=" * 60)
        print("QWEN REQUEST")
        print("=" * 60)

        print(
            f"Prompt characters: {len(prompt)}"
        )

        # ==================================================
        # CREATE TEXT CONTENT
        # ==================================================

        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]


        # ==================================================
        # ADD IMAGES
        # ==================================================

        if image_paths:

            print("\n")
            print("=" * 60)
            print("IMAGES BEING SENT TO QWEN")
            print("=" * 60)

            for image_path in image_paths:

                # --------------------------------------------------
                # Skip empty paths
                # --------------------------------------------------

                if not image_path:

                    print(
                        "⚠️ Empty image path skipped."
                    )

                    continue


                # --------------------------------------------------
                # Convert to Path
                # --------------------------------------------------

                path = Path(
                    image_path
                )


                # --------------------------------------------------
                # Check whether image exists
                # --------------------------------------------------

                if not path.exists():

                    print(
                        f"❌ Image does not exist: {image_path}"
                    )

                    continue


                # --------------------------------------------------
                # Encode image
                # --------------------------------------------------

                base64_image = encode_image(
                    str(path)
                )


                # --------------------------------------------------
                # Detect MIME type
                # --------------------------------------------------

                mime_type = get_image_mime_type(
                    str(path)
                )


                # --------------------------------------------------
                # Add image to multimodal request
                # --------------------------------------------------

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{base64_image}"
                            )
                        }
                    }
                )


                # --------------------------------------------------
                # Debug information
                # --------------------------------------------------

                print(
                    f"✅ EXACT IMAGE: {image_path}"
                )

                print(
                    f"   Size: {path.stat().st_size:,} bytes"
                )

                print(
                    f"   MIME: {mime_type}"
                )


        else:

            print("\n")
            print(
                "No images supplied to Qwen."
            )


        # ==================================================
        # COUNT IMAGES
        # ==================================================

        image_count = sum(
            1
            for item in content
            if item.get("type") == "image_url"
        )


        # ==================================================
        # CALL GROQ / QWEN
        # ==================================================

        print("\n")
        print("=" * 60)
        print("CALLING GROQ / QWEN")
        print("=" * 60)

        print(
            f"Model: {MODEL_NAME}"
        )

        print(
            f"Images: {image_count}"
        )

        print(
            f"Max completion tokens: "
            f"{MAX_COMPLETION_TOKENS}"
        )


        response = client.chat.completions.create(

            # --------------------------------------------------
            # Model
            # --------------------------------------------------

            model=MODEL_NAME,


            # --------------------------------------------------
            # Multimodal message
            # --------------------------------------------------

            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],


            # --------------------------------------------------
            # IMPORTANT
            #
            # Disable Qwen reasoning.
            #
            # This prevents the reasoning process from consuming
            # the available completion-token budget.
            # --------------------------------------------------

            reasoning_effort="none",


            # --------------------------------------------------
            # Hide reasoning from returned output
            # --------------------------------------------------

            reasoning_format="hidden",


            # --------------------------------------------------
            # Output token limit
            # --------------------------------------------------

            max_completion_tokens=MAX_COMPLETION_TOKENS,


            # --------------------------------------------------
            # Sampling
            # --------------------------------------------------

            temperature=0.7,

            top_p=0.8
        )


        # ==================================================
        # READ RESPONSE
        # ==================================================

        print("\n")
        print("=" * 60)
        print("QWEN RESPONSE RECEIVED")
        print("=" * 60)


        # --------------------------------------------------
        # Check choices
        # --------------------------------------------------

        if not response.choices:

            print(
                "❌ Qwen returned no choices."
            )

            return (
                "Qwen did not return a response."
            )


        # --------------------------------------------------
        # Get message
        # --------------------------------------------------

        message = response.choices[0].message


        print(
            f"Message object: {message}"
        )


        # --------------------------------------------------
        # Get content
        # --------------------------------------------------

        answer = message.content


        print(
            f"Content type: {type(answer)}"
        )

        print(
            f"Content length: "
            f"{len(answer) if answer else 0}"
        )


        # ==================================================
        # HANDLE EMPTY RESPONSE
        # ==================================================

        if not answer:

            print(
                "⚠️ Qwen returned empty content."
            )


            # --------------------------------------------------
            # Check whether reasoning exists
            # --------------------------------------------------

            reasoning = getattr(
                message,
                "reasoning",
                None
            )


            if reasoning:

                print(
                    "⚠️ Reasoning was returned "
                    "but final content was empty."
                )

                print(
                    f"Reasoning length: "
                    f"{len(reasoning)}"
                )


            # --------------------------------------------------
            # Check finish reason
            # --------------------------------------------------

            finish_reason = (
                response.choices[0].finish_reason
            )

            print(
                f"Finish reason: "
                f"{finish_reason}"
            )


            return (
                "Qwen returned an empty response."
            )


        # ==================================================
        # REMOVE THINKING TAGS
        # ==================================================

        answer = remove_thinking(
            answer
        )


        # ==================================================
        # FINAL EMPTY CHECK
        # ==================================================

        if not answer:

            print(
                "⚠️ Answer became empty after "
                "removing thinking content."
            )

            return (
                "Qwen returned an empty response."
            )


        # ==================================================
        # FINAL ANSWER
        # ==================================================

        print("\n")
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(
            answer
        )

        print("=" * 60)


        return answer


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as e:

        error_message = str(
            e
        ).lower()


        # ==================================================
        # AUTHENTICATION ERROR
        # ==================================================

        if (
            "401" in error_message
            or "authentication" in error_message
            or "api key" in error_message
            or "invalid api key" in error_message
        ):

            print("\n")
            print("=" * 60)
            print("❌ GROQ API KEY ERROR")
            print("=" * 60)

            print(
                "Your Groq API key may be "
                "expired or invalid."
            )

            print(
                "Please update GROQ_API_KEY "
                "in your .env file."
            )

            print("=" * 60)


            return (
                "LLM API key is invalid or expired. "
                "Please update the API key."
            )


        # ==================================================
        # RATE LIMIT ERROR
        # ==================================================

        if (
            "rate_limit" in error_message
            or "rate limit" in error_message
            or "tokens per minute" in error_message
            or "request too large" in error_message
        ):

            print("\n")
            print("=" * 60)
            print("❌ GROQ RATE LIMIT / TOKEN ERROR")
            print("=" * 60)

            print(
                f"Error: {e}"
            )

            print("=" * 60)


            return (
                "The request is too large for the "
                "current Groq token limit. "
                "Please reduce the amount of document "
                "context or images being sent."
            )


        # ==================================================
        # GENERAL GROQ ERROR
        # ==================================================

        print("\n")
        print("=" * 60)
        print("❌ GROQ API ERROR")
        print("=" * 60)

        print(
            f"Error: {e}"
        )

        print("=" * 60)


        return (
            "The LLM service is currently unavailable."
        )