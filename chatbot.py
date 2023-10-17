from happytransformer import HappyGeneration

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-125M")

from happytransformer import GENSettings

top_k_sampling_settings = GENSettings(do_sample = True, early_stopping = False, top_k = 50, temperature = 0.7, max_length = 300, no_repeat_ngram_size=2)

from happytransformer import GENTrainArgs

args = GENTrainArgs(num_train_epochs=1)

happy_gen.train("chatbot_text.txt", args=args)

while(True):
    x = input("Type your question here...")
    print(happy_gen.generate_text(x, args = top_k_sampling_settings).text)