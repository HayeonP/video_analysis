"""Small procedural video fixture for exercising the real pipeline."""
from pathlib import Path

import av
from PIL import Image, ImageDraw


def make_demo(path: Path, duration: float = 6, fps: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for index in range(round(duration * fps)):
            time = index / fps
            progress = min(1, max(0, (time - duration / 3) / (duration / 3)))
            image = Image.new("RGB", (640, 360), "#ede7d8")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 295, 640, 360), fill="#9c856b")
            draw.rectangle((365, 248, 570, 262), fill="#785334")
            draw.rectangle((380, 260, 392, 335), fill="#785334")
            draw.rectangle((545, 260, 557, 335), fill="#785334")
            # A red-clothed character reaches for and lifts a blue box.
            draw.ellipse((225, 48, 297, 120), fill="#f2bb86", outline="#303039", width=3)
            draw.pieslice((225, 40, 297, 105), 180, 360, fill="#24242b")
            draw.ellipse((265, 76, 271, 83), fill="black")
            draw.line((270, 98, 283, 98), fill="#653523", width=2)
            draw.polygon([(238, 119), (286, 119), (304, 230), (221, 230)], fill="#ba3030")
            draw.line((242, 230, 229, 321), fill="#263b62", width=17)
            draw.line((281, 230, 302, 321), fill="#263b62", width=17)
            draw.line((241, 140, 209, 214), fill="#f2bb86", width=13)
            box_x, box_y = 392 - 55 * progress, 224 - 63 * progress
            draw.line((280, 140, 322, 180, box_x + 8, box_y + 15), fill="#f2bb86", width=13)
            draw.rectangle((box_x, box_y, box_x + 42, box_y + 24), fill="#287ccb", outline="#123855", width=2)
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
