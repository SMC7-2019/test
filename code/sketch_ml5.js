var maxi = maximilian();

var osc_carrier = new maxi.maxiOsc();
var maxiEngine = new maxi.maxiAudio();
maxiEngine.init();

var hz;

let video;
let poseNet;
let poses = [];
let options = {
    imageScaleFactor: 0.5,
    outputStride: 8,
    minConfidence: 0.3,
    maxPoseDetections: 1,
    detectionType: 'single',
    multiplier: 0.75,
}

let playFrom = 70;

function setup() {

    createCanvas(480, 360);

    video = createVideo(['video/Franco Battiato - Centro di gravità permanente 1 - 360.mp4']);
    video.onended(videoEnded);
    video.hide();

    poseNet = ml5.poseNet(video, options, modelReady);
    poseNet.on('pose', function(results) {
        poses = results;
    });

}


function modelReady() {
    video.play().time(playFrom);
}


function videoEnded() {
    video.play().time(playFrom);
}


function draw() {

    image(video, 0, 0, width, height);
    if (poses.length > 0) {
        drawSkeleton();
    }

}


function drawSkeleton() {
    stroke(255);
    strokeWeight(2);
    let skeleton = poses[0].skeleton;
    for (let j = 0; j < skeleton.length; j++) {
        if (j == 2) {
            hz = map(skeleton[j][1].position.x, 0, width, 440, 3520);
            document.getElementById("mousepos").innerHTML = hz.toFixed(2) + "Hz";
            circle(skeleton[j][1].position.x, skeleton[j][1].position.y, 10);
        }
        let partA = skeleton[j][0];
        let partB = skeleton[j][1];
        line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
}


maxiEngine.play = function() {

    output = osc_carrier.sinewave(hz);
    return output * 0.75;

}