"""Join the x64 shards' JUnit reports into one file (tests.yml).

    python3 .github/scripts/join_junit.py SHARDS_DIR OUT.xml

virgil-validation reads one x64 report from main,
``virgil-evidence-ubuntu-latest-py3.12-x641``, which this builds.
"""

import glob
import sys
import xml.etree.ElementTree as ET

joined = ET.Element("testsuites")
for path in sorted(glob.glob(f"{sys.argv[1]}/*/junit-evidence.xml")):
    root = ET.parse(path).getroot()
    joined.extend(root if root.tag == "testsuites" else [root])
ET.ElementTree(joined).write(sys.argv[2], encoding="utf-8", xml_declaration=True)
print(len(joined), "suites")
