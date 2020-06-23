//Front-end of the Chrome extension.
//Uses Javascript to interact with the Stack Overflow website.

//The entire area around the title box, where buttons are added.
var full_title_area = document.getElementById('post-title')
//The actual titlebox.
var title = document.getElementById('title')

//Create the two buttons.
var btn1 = document.createElement("BUTTON")
btn1.innerHTML = "Evaluate title"
btn1.type='button'
btn1.id='iamabutton1'
full_title_area.appendChild(btn1)
var btn2 = document.createElement("BUTTON")
btn2.innerHTML = "Suggest a title"
btn2.type='button'
btn1.id='iamabutton2'
full_title_area.appendChild(btn2)

//The line of text that communicates with the user. Starts out blank.
var predict_line = document.createTextNode(" ");
full_title_area.appendChild(predict_line)

//The textbox where the user edits the body of their question.
var body = document.getElementById('wmd-input')

//What to do when the 'Evaluate title' button is clicked.
function evaluate() {
	console.log(' Evaluating title...')
	if (title.value){
		//Send a request to my Flask webapp (hosted on AWS) to evaluate the quality of the title.
		$.ajax({
			type : 'POST',
			url : 'https://www.titlewave.xyz/evaluate',
			dataType : 'json',
			data : JSON.stringify({'title' : title.value}),
			contentType: "application/json",
			success : function(result) {
				//Update the line of text below the title to report the results.
				predict_line.textContent = " Probability of getting an answer: " + result
			},
			error: function(xhr, status, error) {
				//If an error occurs, print it to the log.
				console.log(xhr.status)
                console.log(status)
            }
		})
	}
	else{
		//If the title field is blank, don't bother sending the request.
		predict_line.textContent = " No title to evaluate"
	}
}

//What to do when the 'Suggest a title' button is clicked.
function suggest() {
	console.log('Suggesting title...')
	predict_line.textContent = ' Thinking...'
	if (body.value){
		//Send a request to my Flask webapp (hosted on AWS) to summarize the question body into a title.
		$.ajax({
			type : 'POST',
			url : 'https://www.titlewave.xyz/suggest',
			dataType : 'json',
			data : JSON.stringify({'body' : body.value}),
			contentType: "application/json",
			success : function(result) {
				//On success, update the title textbox with the suggested title.
				console.log(result)
				title.value = result
				evaluate()
			},
			error: function(xhr, status, error) {
				predict_line.textContent('Error')
				console.log(xhr.status)
                console.log(status)
            }
		})
	}
	else{
		predict_line.textContent = " No question body to summarize"
	}
}


//Add those two functions to the buttons as on-click events.
btn1.onclick = evaluate
btn2.onclick = suggest