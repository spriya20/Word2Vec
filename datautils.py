from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

# import nltk

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text()
    review_text = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      review_text)
    words = review_text.lower().split()
    stops = set(stopwords.words("english"))
    imp_words = [w for w in words if w not in stops]
    return " ".join(imp_words)

