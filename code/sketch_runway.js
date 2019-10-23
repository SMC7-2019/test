var maxi = maximilian();

var osc_carrier = new maxi.maxiOsc();
var maxiEngine = new maxi.maxiAudio();
maxiEngine.init();

var hz = 440;


function setup() {

    createCanvas(480, 360);
    fill(128);
    noStroke();

    frameRate(2);

}


function draw() {

    background(240);

    fetch('http://localhost:8001/data')
        .then(response => response.json())
        .then(outputs => {
            const { points, labels } = outputs;
            drawFace(points);
        })
}


function drawFace(points) {
    xmed = 0;
    for (let j = 0; j < points.length; j++) {
        xmed += points[j][0];
        circle(points[j][0] * width, points[j][1] * height, 5);
    }
    //console.log(points[j][0]);
    xmed /= points.length;
    hz = map(xmed, 0, 1, 440, 3520);
    document.getElementById("facepos").innerHTML = hz.toFixed(2) + "Hz";    
}


maxiEngine.play = function() {

    output = osc_carrier.sinewave(hz);
    return output * 0.75;

}