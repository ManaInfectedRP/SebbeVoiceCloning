"""
Download a static ffmpeg.exe for Windows into the project root.
Run this once before building the .exe with PyInstaller.
"""
import os
import sys
import zipfile
import urllib.request

FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip"
)
DEST = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")


def main():
    if os.path.exists(DEST):
        print(f"ffmpeg.exe already present at {DEST}")
        return

    zip_path = DEST.replace(".exe", ".zip")
    print(f"Downloading ffmpeg from:\n  {FFMPEG_URL}")
    print("(This is ~75MB — please wait...)")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(FFMPEG_URL, zip_path, reporthook=_progress)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)

    print("\nExtracting ffmpeg.exe...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find ffmpeg.exe inside the nested folder
        target = next(
            (n for n in zf.namelist() if n.endswith("/bin/ffmpeg.exe")),
            None
        )
        if not target:
            print("ERROR: ffmpeg.exe not found in zip archive.")
            os.unlink(zip_path)
            sys.exit(1)
        with zf.open(target) as src, open(DEST, "wb") as dst:
            dst.write(src.read())

    os.unlink(zip_path)
    print(f"ffmpeg.exe saved to {DEST}")


if __name__ == "__main__":
    main()
