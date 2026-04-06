import requests
import json
import os

import dotenv

dotenv.load_dotenv()

response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer {os.getenv('OPEN_ROUTER_KEY')}"
  }
)

print(json.dumps(response.json(), indent=2))
