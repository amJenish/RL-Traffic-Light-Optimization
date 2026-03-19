import xml.etree.ElementTree as ET
root = ET.parse('src/data/sumo/flows/flows_day_00.rou.xml').getroot()
flows = root.findall('flow')[:6]
for f in flows:
    print(f"from={f.get('from')}  to={f.get('to')}")