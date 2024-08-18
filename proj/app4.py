from streamlit_webrtc import webrtc_streamer , RTCConfiguration, VideoTransformerBase
import av
import cv2

def callback(frame:av.VideoFrame)->av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        img = cv2.Canny(img , 100, 200)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return av.VideoFrame.from_ndarray(img , format="bgr24")

webrtc_streamer(key="sample" , video_frame_callback=callable,media_stream_constraints={"video": True, "audio": False})
