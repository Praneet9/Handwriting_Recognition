function init() {
  var canvas = document.getElementById('canvas');
  console.log(canvas);
  
	var context = canvas.getContext("2d");
	canvas.width = 280;
	canvas.height = 280;

	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};
	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 7;
    context.lineJoin = context.lineCap = 'round';
	
	debug();

	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;
		// console.log("offset");
		
		// console.log(this.offsetLeft);
		// console.log(this.offsetTop);
		// console.log("pagex");
		
		// console.log(e.pageX);
		// console.log(e.pageY);
		// console.log("mouse");
		
		// console.log(Mouse.x);
		// console.log(Mouse.y);
		
		
		
		
		
		Mouse.x = e.pageX - this.offsetLeft;
		Mouse.y = e.pageY - this.offsetTop;
	}, false);

	canvas.addEventListener("mousedown", function(e) {
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);

	var onPaint = function() {	
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
	
		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();
	};

	function debug() {
		$("#clearButton").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});
	}
}

init();

$(".myButton").click(function(){
	$('#result').text('  Predicting...');
	var canvasObj = document.getElementById("canvas");
	var img = canvasObj.toDataURL('image/png');
	// console.log(img);
	
	$.ajax({
		type: "POST",
		url: "/predict",
		data: img,
		success: function(data){
			$('#result').text('Predicted Output: ' + data);
		}
	});
});