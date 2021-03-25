import pandas as pd

sentence = 'The King of England'
sentence_2 = 'Lets go to England'
sentence_3 = 'He is the new King'
sentence_4 = 'I love eating bananas'
sentence_5 = 'He is a king'
sentence_6 = 'I lived in england for almost all my life'
sentence_7 = 'My name is Alexa'

df = pd.DataFrame({'Sentence' :  [sentence, sentence_2, sentence_3, sentence_4, sentence_5, sentence_6, sentence_7]})

print(df)
