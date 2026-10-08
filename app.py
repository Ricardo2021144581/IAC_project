"""IML project: emotion from facial landmarks, as a web application.

Run with
    python app.py
and open the address printed in the terminal (by default http://127.0.0.1:7860).

The page records a short clip with the camera (or accepts a video file), extracts the
68 facial landmarks of every frame (landmarks.py), and shows the probabilities returned
by predict() in classifier.py. Only classifier.py is to be changed.
"""
import os
import tempfile

import gradio as gr

import classifier
from landmarks import extract_landmarks

MIN_FRAMES = 30   # at least one second of video with a face

model = classifier.load_model()


def run(video_path):
    if not video_path:
        raise gr.Error("Record a clip or upload a video first.")
    frames, info, preview = extract_landmarks(video_path)
    if len(frames) < MIN_FRAMES:
        raise gr.Error(f"A face was found in too few frames ({info['frames_with_face']} of {info['frames_read']}). "
                       "Face the camera, with good light, and record again.")
    scores = classifier.predict(model, frames)
    scores = {str(k): float(v) for k, v in scores.items()}
    csv_path = os.path.join(tempfile.mkdtemp(), "landmarks.csv")
    frames.to_csv(csv_path, index=False)
    details = (f"**Recording.** {info['frames_read']} frames read, face found in {info['frames_with_face']}; "
               f"{info['duration_s']} s at about {info['fps_measured']} frames per second, {info['resolution']} pixels. "
               f"Resampled to {info['frames_at_30fps']} frames at 30 frames per second. Detector: {info['detector']}.")
    return scores, preview, details, csv_path


with gr.Blocks(title="IML: emotion from facial landmarks") as demo:
    gr.Markdown("## Emotion from facial landmarks\n"
                "Record a clip of 3 to 5 seconds, facing the camera, while saying "
                "*“Kids are talking by the door”* with an emotion. Then press **Classify**.")
    with gr.Row():
        video = gr.Video(sources=["webcam", "upload"], label="Clip", height=360)
        with gr.Column():
            label = gr.Label(num_top_classes=8, label="Prediction")
            details = gr.Markdown()
    button = gr.Button("Classify", variant="primary")
    with gr.Row():
        preview = gr.Image(label="68 landmarks, middle frame", height=360)
        table = gr.File(label="Landmarks of the clip (CSV, in the format of the dataset)")
    button.click(run, inputs=video, outputs=[label, preview, details, table])

if __name__ == "__main__":
    demo.launch()
