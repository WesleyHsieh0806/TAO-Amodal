from contextlib import contextmanager

from pathlib import Path

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.tools import extensions_dict


@contextmanager
def video_writer(output,
                 size,
                 fps=30,
                 codec=None,
                 ffmpeg_params=None):
    """
    Args:
        size (tuple): (width, height) tuple
    """
    if isinstance(output, Path):
        output = str(output)

    if codec is None:
        extension = Path(output).suffix[1:]
        try:
            codec = extensions_dict[extension]['codec'][0]
        except KeyError:
            raise ValueError(f"Couldn't find the codec associated with the "
                             f"filename ({output}). Please specify codec")

    if ffmpeg_params is None:
        ffmpeg_params = [
            '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", '-pix_fmt', 'yuv420p'
        ]
    with FFMPEG_VideoWriter(output,
                            size=size,
                            fps=fps,
                            codec=codec,
                            ffmpeg_params=ffmpeg_params) as writer:
        yield writer


def video_info(video):
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
    if isinstance(video, Path):
        video = str(video)

    info = ffmpeg_parse_infos(video)
    return {
        'duration': info['duration'],
        'fps': info['video_fps'],
        'size': info['video_size']  # (width, height)
    }
