from __future__ import annotations

import sys

from tithing_video_maker.app import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--ui", *sys.argv[1:]]
    main()
