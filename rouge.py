from pythonrouge.pythonrouge import Pythonrouge

# # system summary(predict) & reference summary
summary = [[[" The capital of Japan the biggest city in the world."]]]
reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]


# only sentence
rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
score = rouge.calc_score()
print(score)



peer_path = "/Users/ryousuke/desktop/nlp/summarization/scientific_paper/system_sum"
model_path = "/Users/ryousuke/desktop/nlp/summarization/scientific_paper/golden_sum"

# is a directory
rouge = Pythonrouge(summary_file_exist=True,
                    peer_path=peer_path, model_path=model_path,
                    n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                    recall_only=True,
                    stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
score = rouge.calc_score()
print(score)