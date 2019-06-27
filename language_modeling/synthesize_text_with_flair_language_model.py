from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

# char_lm_embeddings = FlairEmbeddings('news-forward')

lm = LanguageModel.load_language_model('flair_resources/language_model/epoch_9.pt')
text, likelihood = lm.generate_text('It is',number_of_characters=100); print(text)
