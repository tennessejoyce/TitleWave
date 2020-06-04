from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def suggest_a_title(body):
	body = 'summarize: ' + body
	input_ids = tokenizer.encode(body, return_tensors="pt")  # Batch size 1
	outputs = model.generate(input_ids,max_length=40,num_beams=10,length_penalty=2)
	summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
	return summary