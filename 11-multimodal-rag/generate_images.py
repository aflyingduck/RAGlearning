"""
Generate two simple diagram images for the multimodal corpus. Pure PIL, no
API calls -- these stand in for product diagrams a real support corpus
would have as actual images (photos, screenshots, PDFs-turned-images)
rather than text.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
IMAGES_DIR = HERE / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def status_ring_legend():
    """A color legend: status ring color -> what it means. This information
    IS in troubleshooting.md/installation-guide.md as text, but here it
    exists ONLY as an image, so retrieval has to actually use the image,
    not a caption standing in for it.
    """
    img = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), "Nimbus Hub — Status Ring Colors", fill="black", font=font(26))

    rows = [
        ("blue", (70, 130, 220), "Booting up"),
        ("white", (235, 235, 235), "Pairing mode"),
        ("green", (40, 160, 70), "Connected, working normally"),
        ("red (solid)", (200, 40, 40), "Lost Wi-Fi connection"),
        ("red (flashing)", (220, 90, 90), "Firmware update failed"),
        ("purple", (140, 60, 170), "Factory reset in progress (Hub 3)"),
        ("orange", (230, 140, 30), "Factory reset in progress (Hub 2)"),
    ]
    y = 80
    for name, color, meaning in rows:
        draw.ellipse([30, y, 80, y + 50], fill=color, outline="black", width=2)
        draw.text((100, y + 12), f"{meaning}", fill="black", font=font(20))
        y += 58

    img.save(IMAGES_DIR / "status-ring-legend.png")


def hub_mesh_topology():
    """A network diagram: two hubs linked via Hub Mesh, each with paired
    devices. Also image-only information.
    """
    img = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), "Hub Mesh — Two-Hub Topology Example", fill="black", font=font(24))

    def hub_box(x, y, label):
        draw.rectangle([x, y, x + 160, y + 70], outline="black", width=3, fill=(225, 235, 250))
        draw.text((x + 15, y + 25), label, fill="black", font=font(18))

    hub_box(60, 120, "Hub 3 (Primary)\n120 devices")
    hub_box(420, 120, "Hub 3 (Secondary)\n95 devices")
    draw.line([220, 155, 420, 155], fill="black", width=3)
    draw.text((250, 130), "Hub Mesh link\n(same LAN required)", fill="black", font=font(14))

    for i in range(3):
        dx = 80 + i * 40
        draw.ellipse([dx, 250, dx + 20, 270], fill=(100, 180, 100), outline="black")
    draw.text((60, 280), "devices", fill="black", font=font(14))

    for i in range(3):
        dx = 440 + i * 40
        draw.ellipse([dx, 250, dx + 20, 270], fill=(100, 180, 100), outline="black")
    draw.text((440, 280), "devices", fill="black", font=font(14))

    draw.text((20, 360), "Total devices across mesh: 215 (well under the 400-device", fill="black", font=font(16))
    draw.text((20, 382), "combined ceiling for a 2-hub mesh at 200 devices/hub each)", fill="black", font=font(16))

    img.save(IMAGES_DIR / "hub-mesh-topology.png")


if __name__ == "__main__":
    status_ring_legend()
    hub_mesh_topology()
    print("Generated:", list(IMAGES_DIR.glob("*.png")))
