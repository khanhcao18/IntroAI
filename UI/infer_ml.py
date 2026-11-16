import re
import string


def remove_punc(text):
    punc = string.punctuation
    return text.translate(str.maketrans('', '', punc))


def remove_emote(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"  # other symbols
        u"\U000024C2-\U0001F251"  # enclosed characters
        u"\U0001f926-\U0001f937"  # gestures
        u"\U0001F1F2"             # specific characters
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"          # gender symbols
        "]+", flags=re.UNICODE)
    
    # Substitute emojis with a space
    text = emoji_pattern.sub(r" ", text)
    return text


def predict_input(user_input, model, vectorizer):
    user_input = remove_emote(remove_punc(user_input.lower()))
    return model.predict(vectorizer.transform([user_input]))
