from flask import Flask, render_template_string
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Face Recognition Web Client</title>
<style>
  body { background: #222; color: #eee; font-family: Arial, sans-serif; text-align: center; margin:0; padding:20px; }
  h1 { margin-bottom: 15px; }

  #container {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 20px;
    max-width: 1360px;
    margin: 0 auto;
  }

  #video, #canvas {
    border-radius: 8px;
    max-width: 640px;
    width: 100%;
    height: auto;
    background: black;
    aspect-ratio: 4 / 3;
  }

  #video {
    -webkit-transform: scaleX(-1);
    transform: scaleX(-1);
  }
                                  
  @media (max-width: 768px) {
      #video {
          width: 100%;
          height: auto;
      }
  }

  #log {
    margin: 20px auto 0 auto;
    max-width: 1360px;
    height: 120px;
    overflow-y: auto;
    background: #111;
    padding: 8px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 0.9em;
    text-align: left;
    white-space: pre-line;
  }
</style>
</head>
<body>
<h1>Face Recognition Web Client</h1>
<div id="container">
  <video id="video" width="640" height="480" autoplay muted></video>
  <canvas id="canvas" width="640" height="480"></canvas>
</div>
<div id="log">Starting...</div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const log = document.getElementById("log");

function logMessage(msg) {
    log.textContent = msg + "\\n" + log.textContent;
}

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        return new Promise(resolve => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });
    } catch (err) {
        logMessage("Error accessing webcam: " + err.message);
        throw err;
    }
}

async function sendFrame() {
    if (video.readyState !== 4) return;

    const offscreen = document.createElement("canvas");
    offscreen.width = video.videoWidth || 640;
    offscreen.height = video.videoHeight || 480;
    const offCtx = offscreen.getContext("2d");
    offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

    const blob = await new Promise(resolve => offscreen.toBlob(resolve, "image/jpeg", 0.8));
    if (!blob) {
        logMessage("Failed to get image blob!");
        return;
    }

    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    try {
        // const apiUrl = `${window.location.protocol}//${window.location.hostname}:5000/recognize-face`;
        // const apiUrl = `https://lightweightfacerecognition.onrender.com/recognize-face`;
        const apiUrl = `https://lwfrdebug.onrender.com/recognize-face`;
        const response = await fetch(apiUrl, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            logMessage("Server error: " + response.status);
            return;
        }

        const data = await response.json();

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw mirrored video frame on canvas (to match video)
        ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height); // ini frame yang dikirim ke server


        if (data.faces && data.faces.length > 0) {
            for (const face of data.faces) {
                const box = face.bounding_box;
                ctx.lineWidth = 3;
                ctx.strokeStyle = face.name === "Unknown" ? "red" : "lime";
                ctx.fillStyle = ctx.strokeStyle;

                const left = box.left;
                const top = box.top;
                const width = box.right - box.left;
                const height = box.bottom - box.top;

                ctx.strokeRect(left, top, width, height);

                let label = face.name;
                if (face.similarity !== null && face.similarity !== undefined) {
                    label += ` (${face.similarity.toFixed(2)})`;
                }

                ctx.font = "18px Arial";
                const textWidth = ctx.measureText(label).width;
                const textHeight = 20;
                let textX = left;
                let textY = top - 10;
                if (textY < textHeight) textY = top + textHeight + 5;

                ctx.fillRect(textX - 2, textY - textHeight + 3, textWidth + 10, textHeight);

                ctx.fillStyle = "black";
                ctx.fillText(label, textX + 5, textY);
            }
            logMessage(`Detected ${data.faces.length} face(s)`);
        } else {
            logMessage("No faces detected");
        }
    } catch (err) {
        logMessage("Error sending frame: " + err.message);
    }
}

async function mainLoop() {
    await sendFrame();
    setTimeout(mainLoop, 500);
}

(async () => {
    try {
        await setupCamera();
        logMessage("Webcam started.");
        mainLoop();
    } catch (e) {
        logMessage("Failed to start webcam.");
    }
})();
</script>
</body>
</html>
    """)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
