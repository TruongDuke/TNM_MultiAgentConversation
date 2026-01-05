import re
import time
import json

from functools import wraps


def prase_json(text):
    flag = False
    
    # Try to extract JSON from Python code (look for json.dumps or dict literals)
    if "```python" in text:
        # Try to find JSON in json.dumps(...)
        json_dumps_match = re.search(r'json\.dumps\((\{.*?\})\)', text, re.DOTALL)
        if json_dumps_match:
            json_str = json_dumps_match.group(1)
            json_data = json.loads(json_str)
            flag = True
        else:
            # Try to find dict literal and convert to JSON
            dict_match = re.search(r'\{[^{}]*"top_k_specialists"[^{}]*\[.*?\]\s*\}', text, re.DOTALL)
            if dict_match:
                # Convert Python dict syntax to JSON (handle single quotes, etc.)
                dict_str = dict_match.group(0)
                dict_str = dict_str.replace("'", '"')  # Replace single quotes
                json_data = json.loads(dict_str)
                flag = True
    
    if not flag:
        if "```json" in text:
            json_match = re.search(r"```json(.*?)```", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                json_data = json.loads(json_str)
                flag = True
        elif "```JSON" in text:
            json_match = re.search(r"```JSON(.*?)```", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                json_data = json.loads(json_str)
                flag = True
        elif "```" in text:
            json_match = re.search(r"```(.*?)```", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                json_data = json.loads(json_str)
                flag = True
        else:
            json_match = re.search(r"{.*?}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
                json_data = json.loads(json_str)
                flag = True
    
    if not flag:
        json_text = text.strip("```json\n").strip("\n```")
        json_data = json.loads(json_text)
    return json_data


def simple_retry(max_attempts=100, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(
                            f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} second..."
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"All {max_attempts} attempts failed. Last error: {str(e)}"
                        )
                        raise

        return wrapper

    return decorator
