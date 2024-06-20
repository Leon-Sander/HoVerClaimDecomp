from transformers import StoppingCriteria, LogitsProcessor


class StopwordLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, stop_words, eos_token_id):
        self.stop_word_sequences = [tokenizer.encode(word, add_special_tokens=False) for word in stop_words]
        self.stop_word_sequences.insert(0, [13, 13, 3100, 28741, 4019, 28747])
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        max_len = max(len(ids) for ids in self.stop_word_sequences)

        for i in range(batch_size):
            input_sequence = input_ids[i, -max_len:].tolist()
            for stop_word_ids in self.stop_word_sequences:
                if len(input_sequence) >= len(stop_word_ids):
                    if input_sequence[-len(stop_word_ids):] == stop_word_ids:
                        # Force the next token to be <eos>
                        scores[i, :] = float('-inf')
                        scores[i, self.eos_token_id] = 0
                        break
        return scores
    
class StopwordCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words):
        self.stop_word_sequences = [tokenizer.encode(word, add_special_tokens=False) for word in stop_words]
        self.stop_word_sequences.insert(0, [13, 13, 3100, 28741, 4019, 28747])
        print("Stop word sequences:", self.stop_word_sequences)  # Debug print
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        max_len = max(len(ids) for ids in self.stop_word_sequences)
        
        for i in range(input_ids.shape[0]):
            input_sequence = input_ids[i, -max_len:].tolist()
            print("Checking sequence:", input_sequence)  # Debug print
            decoded_input = self.tokenizer.decode(input_sequence)
            print("Decoded sequence:", repr(decoded_input))  # Debug print

            for stop_word_ids in self.stop_word_sequences:
                if len(input_sequence) >= len(stop_word_ids):
                    print("comparing: ", input_sequence[-len(stop_word_ids):], stop_word_ids)  # Debug print
                    print("comparing decoded: ", repr(self.tokenizer.decode(input_sequence[-len(stop_word_ids):])), "|||" , repr(self.tokenizer.decode(stop_word_ids)))

                    if input_sequence[-len(stop_word_ids):] == stop_word_ids:
                        print(f"Match found: {stop_word_ids}")  # Debug print
                        return True
        return False