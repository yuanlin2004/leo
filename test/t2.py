import json, os, requests
from time import time
from openai import OpenAI

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# You can use any model that supports tool calling
MODEL = "google/gemini-3-flash-preview"

openai_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

task = "What are the titles of some Yuan Lin books?"
#task = "Hello"

messages = [
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": task,
  }
]

def search_gutenberg_books(search_terms):
    search_query = " ".join(search_terms)
    url = "https://gutendex.com/books"
    response = requests.get(url, params={"search": search_query})

    simplified_results = []
    for book in response.json().get("results", []):
        simplified_results.append({
            "id": book.get("id"),
            "title": book.get("title"),
            "authors": book.get("authors")
        })

    return simplified_results

tools = [
  {
    "type": "function",
    "function": {
      "name": "search_gutenberg_books",
      "description": "Search for books in the Project Gutenberg library based on specified search terms",
      "parameters": {
        "type": "object",
        "properties": {
          "search_terms": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "List of search terms to find books in the Gutenberg library (e.g. ['dickens', 'great'] to search for books by Dickens with 'great' in the title)"
          }
        },
        "required": ["search_terms"]
      }
    }
  }
]

TOOL_MAPPING = {
    "search_gutenberg_books": search_gutenberg_books
}


iter = 0

while iter < 5:
    iter += 1

    # call llm
    request_1 = {
        "model": "google/gemini-3-flash-preview",
        "tools": tools,
        "messages": messages
    }

    #response_1 = openai_client.chat.completions.create(**request_1)
    #print("LLM response with tool call:", response_1.model_dump_json(indent=2))

    # check if the chat completion fails due to rate limits. Catch the exception and wait
    # for a seconds and retry. 
    
    while True:
        try:
            response_1 = openai_client.chat.completions.create(**request_1).choices[0].message
            break
        except Exception as e:
            print(f"LLM call failed with error: {str(e)}. Retrying in 1 seconds...")
            time.sleep(1)
            continue
    #response_1 = openai_client.chat.completions.create(**request_1).choices[0].message
    #messages.append(response_1)

    if not response_1.tool_calls:
        print("Final response from LLM:", response_1.content)
        break
    
    # get tool response and add to messages
    for tool_call in response_1.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        if tool_name in TOOL_MAPPING:
            result = TOOL_MAPPING[tool_name](**tool_args)
            print(f"Result from {tool_name}:", result)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    
