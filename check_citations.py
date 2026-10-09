"""Validate report citations using only the Python standard library."""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REFERENCES_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_CITATION = re.compile(r"\[(\d+(?:\s*(?:,|[–-])\s*\d+)*)\](?![ \t]*\()")
_REFERENCE_LINE = re.compile(r"^\[(\d+)\](?:[ \t]+(.*))?\s*$")
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def _citation_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if not span:
            numbers.append(int(part))
            continue
        start, end = (int(value) for value in span.groups())
        if start <= end and end - start <= 200:
            numbers.extend(range(start, end + 1))
        else:
            numbers.extend((start, end))
    return numbers


def _body_citations(body):
    found = set()
    for index, segment in enumerate(_CODE.split(body)):
        if index % 2:
            continue
        for match in _CITATION.finditer(segment):
            found.update(_citation_numbers(match.group(1)))
    return found


def _reference_urls(text):
    return [match.group(0).rstrip(".,;:!?") for match in _URL.finditer(text or "")]


def check(report_text, sources):
    """Return citation and reference inconsistencies (an empty list means valid)."""
    problems = []
    if not isinstance(sources, list):
        return ["sources.json must contain a JSON list"]
    if not sources:
        return ["no sources in sources.json"]

    by_number = {}
    urls_seen = {}
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} must be an object")
            continue
        number = source.get("n")
        if type(number) is not int:
            problems.append(f"source entry {index}: n must be an integer")
        elif number in by_number:
            problems.append(f"source [{number}] appears more than once in sources.json")
        else:
            by_number[number] = source

        url = source.get("url")
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
            problems.append(f"source [{number}]: url must start with http:// or https://")
        elif url in urls_seen:
            problems.append(
                f"source [{number}] duplicates URL from source [{urls_seen[url]}]"
            )
        else:
            urls_seen[url] = number

    report_text = str(report_text or "")
    headings = list(_REFERENCES_HEADING.finditer(report_text))
    if not headings:
        problems.append("missing ## References heading")
        body = report_text
        references = ""
    else:
        if len(headings) > 1:
            problems.append("multiple ## References headings")
        body = report_text[: headings[0].start()]
        references = report_text[headings[0].end() :]

    cited = _body_citations(body)
    for number in sorted(cited - set(by_number)):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(set(by_number) - cited):
        problems.append(f"source [{number}] never cited")

    reference_counts = {}
    for line_number, line in enumerate(references.splitlines(), start=1):
        match = _REFERENCE_LINE.match(line.strip())
        if not match:
            continue
        number = int(match.group(1))
        reference_counts[number] = reference_counts.get(number, 0) + 1
        if number not in by_number:
            problems.append(f"reference [{number}] has no matching source")
        elif reference_counts[number] > 1:
            problems.append(f"reference [{number}] appears more than once")

        urls = _reference_urls(match.group(2) or "")
        if len(urls) != 1:
            problems.append(
                f"reference [{number}] must contain exactly one http(s) URL (found {len(urls)})"
            )
        elif number in by_number:
            expected = by_number[number].get("url")
            if isinstance(expected, str) and urls[0] != expected:
                problems.append(
                    f"reference [{number}] URL does not match sources.json"
                )

    for number in sorted(set(by_number) - set(reference_counts)):
        problems.append(f"source [{number}] has no References line")
    return problems


def main(argv):
    report_path = argv[0] if len(argv) > 0 else REPORT
    sources_path = argv[1] if len(argv) > 1 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as handle:
            report = handle.read()
        with open(sources_path, encoding="utf-8") as handle:
            sources = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
