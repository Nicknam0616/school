from openai import OpenAI
from deep_translator import GoogleTranslator

# 로컬 서버 설정
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 번역기 설정
translator = GoogleTranslator(source='ko', target='en')

while True:
    user_input = input("AI에게 질문할 문장을 입력하세요 (종료하려면 'q' 입력): ")
    if user_input.lower() == 'q':
        break

    # 한국어에서 영어로 번역
    translated_input = translator.translate(user_input)

    completion = client.chat.completions.create(
        model="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF/llama-3.2-3b-instruct-q8_0.gguf",
        messages=[
            {"role": "system", "content": "You're an expert in strategy and tactics, help me ..."},
            {"role": "user", "content": translated_input}
        ],
        temperature=0.7,
    )

    # AI의 응답을 영어에서 한국어로 번역
    translated_output = GoogleTranslator(source='en', target='ko').translate(completion.choices[0].message.content)

    print("AI의 응답: ", translated_output)

print("프로그램이 종료되었습니다.")
