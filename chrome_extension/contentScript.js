var full_above_title = document.getElementById('post-title')
var above_title = full_above_title.getElementsByTagName('P')[0]
var title = document.getElementById('title')

var btn1 = document.createElement("BUTTON")
btn1.innerHTML = "Evaluate title"
btn1.type='button'
btn1.id='iamabutton1'
full_above_title.appendChild(btn1)
var btn2 = document.createElement("BUTTON")
btn2.innerHTML = "Suggest a title"
btn2.type='button'
btn1.id='iamabutton2'
full_above_title.appendChild(btn2)

var predict_line = document.createTextNode("Hello World");
full_above_title.appendChild(predict_line)

var body = document.getElementById('wmd-input')

btn1.onclick = function() {
	console.log('Evaluating title...')
	if (title.value){
		$.ajax({
			type : 'POST',
			url : 'http://localhost:5000/evaluate',
			dataType : 'json',
			data : JSON.stringify({'title' : title.value}),
			contentType: "application/json",
			success : function(result) {
				predict_line.textContent = "Probability of getting an answer: " + result
			},
			error: function(xhr, status, error) {
				console.log(xhr.status)
                console.log(status)
            }
		})
	}
	else{
		predict_line.textContent = "No title to evaluate"
	}
}


btn2.onclick = function() {
	console.log('Suggesting title...')
	if (body.value){
		$.ajax({
			type : 'POST',
			url : 'http://localhost:5000/suggest',
			dataType : 'json',
			data : JSON.stringify({'body' : body.value}),
			contentType: "application/json",
			success : function(result) {
				console.log(result)
				title.value = result
			},
			error: function(xhr, status, error) {
				console.log(xhr.status)
                console.log(status)
            }
		})
	}
	else{
		predict_line.textContent = "No question body to summarize"
	}
}