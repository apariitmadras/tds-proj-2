import requests

file_path = r"C:\Users\shara\OneDrive\Desktop\shiva\tds\question.txt"

# Use the correct endpoint, for example:
url = "http://localhost:8000/api/analyze"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(f"Status code: {response.status_code}")
    print("Response:")
    print(response.text)