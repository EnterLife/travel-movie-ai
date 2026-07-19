from datetime import UTC, datetime

import pytest

from travelmovieai.infrastructure.ffmpeg import parse_probe_payload


def test_parse_probe_payload_extracts_video_metadata_and_location() -> None:
    result = parse_probe_payload(
        {
            "format": {
                "duration": "12.5",
                "format_name": "mov,mp4",
                "bit_rate": "1000000",
                "tags": {
                    "creation_time": "2026-05-10T11:12:13Z",
                    "com.apple.quicktime.location.ISO6709": "+55.7558+037.6173/",
                },
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "duration": "12.25",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }
    )

    assert result.duration_seconds == 12.5
    assert result.video_duration_seconds == 12.25
    assert result.metadata["video_duration_seconds"] == 12.25
    assert result.width == 1920
    assert result.height == 1080
    assert result.fps == pytest.approx(29.97, rel=0.001)
    assert result.created_at == datetime(2026, 5, 10, 11, 12, 13, tzinfo=UTC)
    assert result.latitude == pytest.approx(55.7558)
    assert result.longitude == pytest.approx(37.6173)


@pytest.mark.parametrize(
    "location",
    ["+091.000+000.000/", "+045.000+181.000/"],
)
def test_parse_probe_payload_ignores_out_of_range_location(location: str) -> None:
    result = parse_probe_payload(
        {
            "format": {
                "duration": "1",
                "tags": {"com.apple.quicktime.location.ISO6709": location},
            },
            "streams": [],
        }
    )

    assert result.latitude is None
    assert result.longitude is None
