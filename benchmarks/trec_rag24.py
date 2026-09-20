"""TREC 2024 RAG, retrieval task: MS MARCO v2.1 segments with NIST relevance labels on a 0-3 scale.

The qrels and topics are public. The segments live in a 27 GB tar of gzipped JSON lines, which is streamed
and never stored: only the judged segments are kept.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

from benchmarks.llmjudge import LLMJudgeData, Split

logger = logging.getLogger(__name__)

NIST = "https://trec.nist.gov/data/rag"
TOPICS_FILE = "topics.rag24.test.txt"
QRELS_FILE = "2024-retrieval-qrels.txt"
CORPUS_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar"
SEGMENTS_DIR = "segments"
# msmarco_v2.1_doc_<file>_<doc offset>#<segment index>_<byte offset of the segment's line in the unzipped file>
SEGMENT_ID = re.compile(r"msmarco_v2\.1_doc_(?P<file>\d+)_\d+#\d+_(?P<offset>\d+)")
TAR_BLOCK = 512
CHUNK = 1 << 20

OpenRange = Callable[[int, int], BinaryIO]


def open_corpus_range(start: int, end: int) -> BinaryIO:
    request = urllib.request.Request(CORPUS_URL, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(request, timeout=120)  # type: ignore[no-any-return]


def tar_members(open_range: OpenRange) -> Iterator[tuple[str, int, int]]:
    """The name, data offset and size of each file in the tar, read from its headers alone."""
    position = 0
    while True:
        header = open_range(position, position + TAR_BLOCK - 1).read()
        name = header[:100].rstrip(b"\0").decode()
        if not name:
            return
        size = int(header[124:136].rstrip(b"\0 ") or b"0", 8)
        if header[156:157] in (b"0", b"\0"):
            yield name, position + TAR_BLOCK, size
        position += TAR_BLOCK + -(-size // TAR_BLOCK) * TAR_BLOCK


def lines_at(gzipped: Iterable[bytes], offsets: Iterable[int]) -> Iterator[bytes]:
    """The lines starting at `offsets` of the unzipped stream, without splitting the rest into lines.

    Stops reading once the last line is complete, so the tail of the stream is never downloaded.
    """
    wanted = sorted(offsets, reverse=True)
    unzip = zlib.decompressobj(wbits=31)
    position = 0
    line = None
    for compressed in gzipped:
        chunk = unzip.decompress(compressed)
        start = 0
        while wanted or line is not None:
            if line is None:
                if wanted[-1] >= position + len(chunk):
                    break
                start = wanted.pop() - position
                line = b""
            end = chunk.find(b"\n", start)
            if end == -1:
                line += chunk[start:]
                break
            yield line + chunk[start:end]
            line, start = None, end
        position += len(chunk)
        if not wanted and line is None:
            return


def extract_member(open_range: OpenRange, data_offset: int, size: int, segments: dict[int, str], target: Path) -> None:
    """The qrels carry a placeholder id, msmarco_v2.1_doc_01_23#45_67, that is no segment of the corpus.
    An id whose offset does not hold that segment is left out, and `load` drops its judgments."""
    stream = open_range(data_offset, data_offset + size - 1)
    found = {}
    for line in lines_at(iter(lambda: stream.read(CHUNK), b""), segments):
        try:
            segment = json.loads(line)
        except json.JSONDecodeError:
            continue
        if segment.get("docid") in segments.values():
            found[segment["docid"]] = {"title": segment["title"], "segment": segment["segment"]}
    missing = set(segments.values()) - found.keys()
    if missing:
        logger.warning(f"Not in corpus file {target.stem}: {', '.join(sorted(missing))}")
    target.write_text(json.dumps(found))


def download(data_dir: Path, open_range: OpenRange = open_corpus_range, n_threads: int = 8) -> None:
    """Resumable: every corpus file writes its judged segments once it is done, and is skipped from then on."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (TOPICS_FILE, QRELS_FILE):
        if not (data_dir / file_name).is_file():
            # trec.nist.gov answers 403 to urllib's default user agent.
            request = urllib.request.Request(f"{NIST}/{file_name}", headers={"User-Agent": "ragelo-benchmarks"})
            (data_dir / file_name).write_bytes(urllib.request.urlopen(request, timeout=120).read())

    by_file: dict[str, dict[int, str]] = {}
    for did in {line.split()[2] for line in (data_dir / QRELS_FILE).read_text().splitlines()}:
        match = SEGMENT_ID.fullmatch(did)
        if match is None:
            raise ValueError(f"Unexpected segment id {did}")
        by_file.setdefault(match["file"], {})[int(match["offset"])] = did

    segments_dir = data_dir / SEGMENTS_DIR
    segments_dir.mkdir(exist_ok=True)
    pending = {number for number in by_file if not (segments_dir / f"{number}.json").is_file()}
    if not pending:
        return
    logger.warning(f"Streaming {len(pending)} of the corpus files for their judged segments. This is a 27 GB tar.")
    with ThreadPoolExecutor(n_threads) as pool:
        jobs = []
        for name, data_offset, size in tar_members(open_range):
            number = name.rsplit("_", 1)[-1].split(".")[0]
            if number in pending:
                target = segments_dir / f"{number}.json"
                jobs.append(pool.submit(extract_member, open_range, data_offset, size, by_file[number], target))
        for job in jobs:
            job.result()


def load(data_dir: Path, split: Split = "test") -> LLMJudgeData:
    """The track has one set of judged topics, so `split` only keeps the signature of the other benchmarks."""
    qrels: dict[str, dict[str, int]] = {}
    for line in (data_dir / QRELS_FILE).read_text().splitlines():
        qid, _, did, label = line.split()
        qrels.setdefault(qid, {})[did] = int(label)

    queries = {}
    for line in (data_dir / TOPICS_FILE).read_text().splitlines():
        qid, text = line.split("\t")
        if qid in qrels:
            queries[qid] = text.strip()

    passages = {}
    for path in sorted((data_dir / SEGMENTS_DIR).glob("*.json")):
        for did, segment in json.loads(path.read_text()).items():
            passages[did] = segment["segment"]
    qrels = {qid: {did: label for did, label in labels.items() if did in passages} for qid, labels in qrels.items()}
    return LLMJudgeData(queries=queries, passages=passages, qrels=qrels)
