// Small helper shared by the verify and enroll pages: starts the webcam and
// exposes a function to grab the current frame as a JPEG data URL.
async function startCamera(videoEl) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    videoEl.srcObject = stream;
    await videoEl.play();
    return stream;
}

function grabFrame(videoEl) {
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.9);
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
