import streamlit as st
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Translate function
def translate(sentence, tokenizer, model, source_lang, target_lang):
    # Set source language
    tokenizer.src_lang = source_lang

    # Encode input
    encoded = tokenizer(sentence, return_tensors = "pt").to(device)

    # Take ID token of target lang
    forced_bos_token_id = tokenizer.get_lang_id(target_lang)

    # Generate
    generated_ids = model.generate(
        **encoded,
        forced_bos_token_id = forced_bos_token_id,
        max_length=128
    )

    # Decode
    translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return translated_text

# Streamlit app
def main():
    st.header('Multilingual Machine Translation')
    st.write('Model: M2M100. Supports over 100 languages')

    sentence = st.text_input("Enter text to translate: ")

    # Mapping languages
    language_dict = {
        "Tiếng Anh": "en",
        "Tiếng Việt": "vi",
        "Tiếng Pháp": "fr",
        "Tiếng Đức": "de",
        "Tiếng Tây Ban Nha": "es",
        "Tiếng Ý": "it",
        "Tiếng Bồ Đào Nha": "pt",
        "Tiếng Nga": "ru",
        "Tiếng Trung": "zh",
        "Tiếng Ả Rập": "ar"
    }

    # Choose source lang + target lang display
    source_lang_display = st.selectbox("Source Language:", list(language_dict.keys()))
    target_lang_display = st.selectbox("Target Language:", list(language_dict.keys()))

    # Take key's language
    source_lang = language_dict[source_lang_display]
    target_lang = language_dict[target_lang_display]

    # Check
    if sentence:
        with st.spinner("Translating..."):
            result = translate(sentence, tokenizer, model, source_lang, target_lang)
        st.success(f'Translated: {result}')

if __name__ == '__main__':
     main()
