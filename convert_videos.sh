#!/usr/bin/env bash
# convert_videos.sh
#
# Converts .webm files to .mp4 (H.264/AAC) or .gif
#
# Usage:
#   ./convert_videos.sh                          # convert all webm in ./demos
#   ./convert_videos.sh demos/                   # convert all webm in a folder
#   ./convert_videos.sh demos/foo.webm           # convert a single file
#   ./convert_videos.sh demos/ --gif             # folder -> gif
#   ./convert_videos.sh demos/foo.webm --gif     # single file -> gif
#
# Requirements: ffmpeg

set -euo pipefail

# ── args ─────────────────────────────────────────────────────────────────────
INPUT_PATH="${1:-demos}"
MODE="${2:---mp4}"

if [[ "$MODE" == "--gif" ]]; then
    TARGET="gif"
else
    TARGET="mp4"
fi

# ── checks ───────────────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found. Install it with:  sudo apt install ffmpeg"
    exit 1
fi

# ── build file list ──────────────────────────────────────────────────────────
shopt -s nullglob

if [[ -f "$INPUT_PATH" ]]; then
    # Single file provided
    if [[ "$INPUT_PATH" != *.webm ]]; then
        echo "ERROR: '$INPUT_PATH' is not a .webm file."
        exit 1
    fi
    FILES=("$INPUT_PATH")
elif [[ -d "$INPUT_PATH" ]]; then
    # Folder provided
    FILES=("$INPUT_PATH"/*.webm)
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "No .webm files found in '$INPUT_PATH'."
        exit 0
    fi
else
    echo "ERROR: '$INPUT_PATH' is not a valid file or folder."
    exit 1
fi

# ── convert ──────────────────────────────────────────────────────────────────
echo "Converting ${#FILES[@]} file(s) -> .$TARGET"
echo "──────────────────────────────────────────────────"

convert_file() {
    local INPUT="$1"
    local BASENAME="${INPUT%.webm}"
    local OUTPUT="${BASENAME}.${TARGET}"

    if [[ -f "$OUTPUT" ]]; then
        echo "SKIP  (already exists): $OUTPUT"
        return
    fi

    echo "Converting: $INPUT -> $OUTPUT"

    if [[ "$TARGET" == "mp4" ]]; then
        # VP8/VP9 -> H.264 video + AAC audio
        # -vsync vfr   : preserve variable frame rate timestamps (fixes Screencastify stutter/skip)
        # -crf 18      : near-lossless quality (lower = better, 18-23 is good range)
        # -preset slow : better compression (change to 'fast' if speed matters)
        ffmpeg -i "$INPUT" \
            -c:v libx264 -crf 18 -preset slow \
            -vsync vfr \
            -c:a aac -b:a 192k \
            -movflags +faststart \
            -y "$OUTPUT" \
            2>&1 | grep -E "^(frame|fps|size|time|bitrate|speed|Error|error)" || true

    elif [[ "$TARGET" == "gif" ]]; then
        # Two-pass gif for best quality with palette optimization
        local PALETTE
        PALETTE=$(mktemp /tmp/palette_XXXX.png)

        # Pass 1: generate palette
        ffmpeg -i "$INPUT" \
            -vf "fps=15,scale=800:-1:flags=lanczos,palettegen=stats_mode=diff" \
            -y "$PALETTE" -loglevel error

        # Pass 2: encode gif using palette
        ffmpeg -i "$INPUT" -i "$PALETTE" \
            -lavfi "fps=15,scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
            -y "$OUTPUT" -loglevel error

        rm -f "$PALETTE"
    fi

    echo "Done:       $OUTPUT"
}

for INPUT in "${FILES[@]}"; do
    convert_file "$INPUT"
done

echo "──────────────────────────────────────────────────"
echo "All conversions complete."
