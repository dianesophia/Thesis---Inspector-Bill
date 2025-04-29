import cv2
import numpy as np
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaPlayer

pcs = set()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Display frame in OpenCV window
        cv2.imshow("WebRTC Stream", img)
        cv2.waitKey(1)

        return frame

async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Receive video from client
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(track))

    await pc.setRemoteDescription(RTCSessionDescription(params["sdp"], params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def cleanup():
    while True:
        await asyncio.sleep(10)
        for pc in list(pcs):
            if pc.connectionState == "closed":
                pcs.discard(pc)

# Start the WebRTC server
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    app.post("/offer")(offer)

    @app.get("/")
    def index():
        return HTMLResponse("<h1>WebRTC Server Running</h1>")

    uvicorn.run(app, host="0.0.0.0", port=8000)
