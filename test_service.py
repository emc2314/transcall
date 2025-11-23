import os
import base64
import io
from PIL import Image
import openai
from google import genai
from google.genai import types

# Configuration
LOCAL_API_URL = "http://localhost:8000"
OPENAI_BASE_URL = f"{LOCAL_API_URL}/v1"
GEMINI_BASE_URL = f"{LOCAL_API_URL}"

# Models to test
MODEL_OPENAI = "gpt-image-1"
MODEL_GEMINI = "gemini-3-pro-image-preview"

def verify_and_return_image(data_bytes: bytes, source_desc: str) -> Image.Image:
    """Verifies the bytes are a valid image and returns the PIL Image object."""
    try:
        img = Image.open(io.BytesIO(data_bytes))
        img.load()  # Force load to verify validity
        print(f"  [PASS] Valid image received from {source_desc} ({img.format} {img.size})")
        return img
    except Exception as e:
        print(f"  [FAIL] Invalid image data from {source_desc}: {e}")
        raise

def run_openai_sdk_tests(target_model: str):
    print(f"\n>>> Testing OpenAI SDK with model: {target_model}")
    client = openai.OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key="dummy-key"
    )

    # 1. Generate
    print(f"  1. Generating image...")
    try:
        resp = client.images.generate(
            model=target_model,
            prompt="A colorful hot air balloon flying over a mountain range",
            size="1024x1024",
            n=1
        )
        if not resp.data:
            print("  [FAIL] No data returned from OpenAI generate.")
            return

        b64_str = resp.data[0].b64_json
        if not b64_str:
             print("  [FAIL] No b64_json in response.")
             return

        img_bytes = base64.b64decode(b64_str)
        generated_img = verify_and_return_image(img_bytes, "Generation")
    except Exception as e:
        print(f"  [ERROR] Generation failed: {e}")
        return

    # 2. Edit (Image + Prompt, no mask)
    print(f"  2. Editing image (converting to black and white)...")
    try:
        # Convert PIL image back to bytes for upload
        img_byte_arr = io.BytesIO()
        generated_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        resp = client.images.edit(
            model=target_model,
            image=img_byte_arr,
            prompt="Make the image black and white, pencil sketch style",
            size="1024x1024",
            n=1
        )
        if not resp.data:
            print("  [FAIL] No data returned from OpenAI edit.")
            return

        b64_str = resp.data[0].b64_json
        if not b64_str:
             print("  [FAIL] No b64_json in edit response.")
             return

        edit_bytes = base64.b64decode(b64_str)
        verify_and_return_image(edit_bytes, "Edit")
    except Exception as e:
        print(f"  [ERROR] Edit failed: {e}")

def run_google_genai_sdk_tests(target_model: str):
    print(f"\n>>> Testing Google GenAI SDK with model: {target_model}")

    client = genai.Client(
        vertexai=True,
        api_key="dummy-key",
        http_options={
            "base_url": GEMINI_BASE_URL,
            "headers": {"Authorization": "Bearer test_token"}
        }
    )

    # 1. Generate
    print(f"  1. Generating image...")
    try:
        response = client.models.generate_content(
            model=target_model,
            contents="A cyberpunk cat wearing sunglasses",
            config=types.GenerateContentConfig(
                response_modalities=[types.Modality.TEXT, types.Modality.IMAGE]
            )
        )

        if not response.candidates:
             print("  [FAIL] No candidates returned.")
             return

        cand = response.candidates[0]
        if not cand.content or not cand.content.parts:
             print("  [FAIL] Candidate has no content or parts.")
             return

        # The unified mapper puts the b64 string into inline_data
        # Iterate through parts to find the image, as thinking models return multiple parts
        image_part = None
        if cand.content and cand.content.parts:
            for part in cand.content.parts:
                if part.inline_data:
                    image_part = part
                    break
        
        if image_part and image_part.inline_data and image_part.inline_data.data:
            # SDK automatically decodes base64 inline_data.data to bytes
            generated_img = verify_and_return_image(image_part.inline_data.data, "Generation")
        else:
            print("  [FAIL] No inline image data found in any part.")
            return

    except Exception as e:
        print(f"  [ERROR] Generation failed: {e}")
        return

    # 2. Edit (Visual Prompting)
    print(f"  2. Editing image (converting to black and white)...")
    try:
        # For visual prompting, we pass the PIL Image directly
        # Casting list contents for MyPy satisfaction (though runtime is fine)
        edit_contents = [generated_img, "Make this image black and white, high contrast"]

        response = client.models.generate_content(
            model=target_model,
            contents=edit_contents  # type: ignore
        )

        if not response.candidates:
             print("  [FAIL] No candidates returned for edit.")
             return

        cand = response.candidates[0]
        if not cand.content or not cand.content.parts:
             print("  [FAIL] Candidate has no content or parts.")
             return

        image_part = None
        for part in cand.content.parts:
            if part.inline_data:
                image_part = part
                break

        if image_part and image_part.inline_data and image_part.inline_data.data:
            verify_and_return_image(image_part.inline_data.data, "Edit")
        else:
            print("  [FAIL] No inline image data found in edit response.")


    except Exception as e:
        print(f"  [ERROR] Edit failed: {e}")

if __name__ == "__main__":
    print("=== Starting Transcall Service Tests (8 Total) ===")

    # 1. OpenAI SDK -> OpenAI Model
    run_openai_sdk_tests(MODEL_OPENAI)

    # 2. OpenAI SDK -> Gemini Model (Cross-Provider)
    run_openai_sdk_tests(MODEL_GEMINI)

    # 3. Google GenAI SDK -> OpenAI Model (Cross-Provider)
    run_google_genai_sdk_tests(MODEL_OPENAI)

    # 4. Google GenAI SDK -> Gemini Model
    run_google_genai_sdk_tests(MODEL_GEMINI)

    print("\n=== All Tests Completed ===")