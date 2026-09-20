import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from benchmarks.trec_rag24 import QRELS_FILE, TOPICS_FILE, download, lines_at, load, tar_members


def corpus_file(number: str, segments: list[str]) -> tuple[bytes, dict[str, int]]:
    """A gzipped corpus file and the byte offset of each segment's line, as the real segment ids carry it."""
    lines, offsets, position = [], {}, 0
    for i, text in enumerate(segments):
        offsets[text] = position
        did = f"msmarco_v2.1_doc_{number}_0#{i}_{position}"
        line = json.dumps({"docid": did, "title": f"Title {i}", "segment": text}).encode() + b"\n"
        lines.append(line)
        position += len(line)
    return gzip.compress(b"".join(lines)), offsets


def tar_of(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class TestStreaming:
    @pytest.mark.parametrize("chunk_size", [1, 7, 4096])
    def test_lines_at_returns_the_lines_at_the_offsets_however_the_stream_is_cut(self, chunk_size):
        gzipped, offsets = corpus_file("00", ["alpha", "beta " * 50, "gamma", "delta"])
        chunks = [gzipped[i : i + chunk_size] for i in range(0, len(gzipped), chunk_size)]

        lines = list(lines_at(chunks, [offsets["gamma"], offsets["beta " * 50]]))

        assert [json.loads(line)["segment"] for line in lines] == ["beta " * 50, "gamma"]

    def test_lines_at_stops_reading_after_the_last_wanted_line(self):
        gzipped, offsets = corpus_file("00", ["alpha"] + ["filler " * 2000] * 50)
        read = []

        def chunks():
            for i in range(0, len(gzipped), 64):
                read.append(i)
                yield gzipped[i : i + 64]

        assert len(list(lines_at(chunks(), [offsets["alpha"]]))) == 1
        assert len(read) < len(gzipped) / 64

    def test_tar_members_reads_names_offsets_and_sizes_from_the_headers(self):
        tar = tar_of({"corpus/seg_07.json.gz": b"x" * 700, "corpus/seg_03.json.gz": b"y" * 10})

        members = list(tar_members(lambda start, end: io.BytesIO(tar[start : end + 1])))

        assert [(name, size) for name, _, size in members] == [
            ("corpus/seg_07.json.gz", 700),
            ("corpus/seg_03.json.gz", 10),
        ]
        assert all(tar[offset : offset + size] in (b"x" * 700, b"y" * 10) for _, offset, size in members)


class TestLoader:
    def test_download_keeps_the_judged_segments_and_load_drops_the_ids_the_corpus_does_not_have(
        self, tmp_path: Path, caplog
    ):
        first, first_offsets = corpus_file("00", ["not judged", "about vicarious trauma"])
        second, second_offsets = corpus_file("01", ["about disability insurance", "also not judged"])
        tar = tar_of(
            {"c/msmarco_v2.1_doc_segmented_01.json.gz": second, "c/msmarco_v2.1_doc_segmented_00.json.gz": first}
        )
        trauma = f"msmarco_v2.1_doc_00_0#1_{first_offsets['about vicarious trauma']}"
        insurance = f"msmarco_v2.1_doc_01_0#0_{second_offsets['about disability insurance']}"
        (tmp_path / TOPICS_FILE).write_text("2024-1\twhat is vicarious trauma?\r\n2024-2\tunjudged topic\r\n")
        placeholder = "msmarco_v2.1_doc_01_23#45_67"
        (tmp_path / QRELS_FILE).write_text(f"2024-1 0 {trauma} 3\n2024-1 0 {insurance} 0\n2024-1 0 {placeholder} 0\n")
        ranges = []

        def open_range(start: int, end: int) -> io.BytesIO:
            ranges.append((start, end))
            return io.BytesIO(tar[start : end + 1])

        download(tmp_path, open_range=open_range, n_threads=2)
        data = load(tmp_path)
        n_requests = len(ranges)
        download(tmp_path, open_range=open_range)

        assert data.queries == {"2024-1": "what is vicarious trauma?"}
        assert data.qrels == {"2024-1": {trauma: 3, insurance: 0}}
        assert data.passages == {trauma: "about vicarious trauma", insurance: "about disability insurance"}
        assert len(ranges) == n_requests
        assert "Not in corpus file 01: msmarco_v2.1_doc_01_23#45_67" in caplog.text
