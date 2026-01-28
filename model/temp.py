import google.generativeai as genai

# Make sure you have configured your API key
genai.configure(api_key="AIzaSyBDuel0CPx4qWnRMCGwpw0PPtSnIZTgeYo")

# List available models
models = genai.list_models()
for m in models:
    print(m.name, getattr(m, "capabilities", None))
