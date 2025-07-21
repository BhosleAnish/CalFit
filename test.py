import google.generativeai as genai

genai.configure(api_key="AIzaSyCcNL6ty0oer-tSZa6ruHKwAt4pcrul8is")  # from aistudio.google.com

model = genai.GenerativeModel("gemini-1.5-pro-latest")

response = model.generate_content("Hello Gemini, test success?")
print(response.text)
