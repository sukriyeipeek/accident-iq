import pandas as pd
import pytest

from scripts.download_stats19 import download_file, extract_years_from_cumulative


class Response:
    def __init__(self, chunks, size):
        self.chunks = chunks
        self.headers = {"Content-Length": str(size)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self.chunks


def test_download_publishes_only_complete_file(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.download_stats19.requests.get", lambda *args, **kwargs: Response([b"ab", b"c"], 3))
    target = tmp_path / "data.csv"
    download_file("https://example.test/data.csv", target)
    assert target.read_bytes() == b"abc"
    assert not target.with_suffix(".csv.part").exists()


def test_truncated_download_never_becomes_final(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.download_stats19.requests.get", lambda *args, **kwargs: Response([b"ab"], 3))
    target = tmp_path / "data.csv"
    with pytest.raises(ValueError, match="Incomplete download"):
        download_file("https://example.test/data.csv", target)
    assert not target.exists()
    assert target.with_suffix(".csv.part").exists()


def test_incomplete_existing_download_is_not_silently_skipped(tmp_path, monkeypatch):
    target = tmp_path / "data.csv"
    target.write_bytes(b"ab")
    monkeypatch.setattr("scripts.download_stats19.requests.get", lambda *args, **kwargs: Response([b"abc"], 3))
    with pytest.raises(ValueError, match="Cannot verify"):
        download_file("https://example.test/data.csv", target)
    assert target.read_bytes() == b"ab"


def test_extract_and_validate_year_counts(tmp_path):
    source = tmp_path / "archive.csv"
    pd.DataFrame({"collision_year": [2019, 2020, 2019], "collision_index": ["a", "b", "c"]}).to_csv(source, index=False)
    extract_years_from_cumulative(source, tmp_path, "collision", range(2019, 2021))
    path = tmp_path / "dft-road-casualty-statistics-collision-2019.csv"
    result = pd.read_csv(path)
    assert len(result) == 2 and result["collision_year"].eq(2019).all()
    result.iloc[:1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagrees with archive"):
        extract_years_from_cumulative(source, tmp_path, "collision", range(2019, 2020))


def test_extract_missing_year_fails_without_final_file(tmp_path):
    source = tmp_path / "archive.csv"
    pd.DataFrame({"collision_year": [2020]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="No rows"):
        extract_years_from_cumulative(source, tmp_path, "collision", range(2019, 2020))
    assert not (tmp_path / "dft-road-casualty-statistics-collision-2019.csv").exists()
