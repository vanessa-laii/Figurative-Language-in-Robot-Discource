"""Video acquisition; runtime/authentication options do not change OCR settings."""

from pathlib import Path

from .common import CACHE


def download_options(cookies=None):
    options = {"format": "bestvideo[height<=1080]/best[height<=1080]", "noplaylist": True,
               "outtmpl": str(CACHE / "media/%(id)s.%(ext)s"), "quiet": True,
               "retries": 3, "fragment_retries": 3, "socket_timeout": 30}
    deno = CACHE / "tools/deno/deno"
    if deno.is_file():
        options["js_runtimes"] = {"deno": {"path": str(deno)}}
    if cookies is not None:
        cookies = Path(cookies).expanduser().resolve()
        if not cookies.is_file():
            raise ValueError("Cookies file does not exist")
        options["cookiefile"] = str(cookies)
    return options


def obtain_video(video_id, cookies=None):
    directory = CACHE / "media"
    directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(directory.glob(f"{video_id}.*")):
        if path.suffix in {".mp4", ".webm", ".mkv", ".mov"}:
            return path
    import yt_dlp

    with yt_dlp.YoutubeDL(download_options(cookies)) as downloader:
        info = downloader.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        path = Path(downloader.prepare_filename(info))
    if not path.is_file():
        raise ValueError("Downloader did not produce a completed video")
    return path
