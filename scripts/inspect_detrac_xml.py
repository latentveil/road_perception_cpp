import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def inspect_xml(xml_path: str) -> None:
	xml_path = Path(xml_path)

	if not xml_path.exists():
		raise FileNotFoundError(f"XML file not found: {xml_path}")

	tree = ET.parse(xml_path)
	root = tree.getroot()

	print("=" * 60)
	print("XML:", xml_path)
	print("root tag:", root.tag)
	print("sequence name:", root.attrib.get("name"))

	# sequence_attribute
	seq_attr = root.find("sequence_attribute")
	if seq_attr is not None:
		print("sequence attribute:", seq_attr.attrib)

	# ignore regions
	ignored_region = root.find("ignored_region")
	ignored_boxes = []

	if ignored_region is not None:
		ignored_boxes = ignored_region.findall("box")

	print("ignored region count:", len(ignored_boxes))

	# frames
	frames = root.findall("frame")
	print("frame count:", len(frames))

	if not frames:
		print("No frames found.")
		return

	# inspect first frame
	first_frame = frames[0]

	print("\n" + "=" * 20 + " FIRST FRAME " + "=" * 20)
	print("frame num:", first_frame.attrib.get("num"))
	print("frame density:", first_frame.attrib.get("density"))

	target_list = first_frame.find("target_list")

	if target_list is None:
		print("No target_list found")
		return

	targets = target_list.findall("target")
	print("target count:", len(targets))

	if not targets:
		print("No targets found.")
		return

	# inspect first target
	first_target = targets[0]

	print("\n" + "=" * 20 + " FIRST TARGET " + "=" * 19)
	print("target id:", first_target.attrib.get("id"))

	box = first_target.find("box")
	if box is not None:
		print("bbox:", box.attrib)

	attr = first_target.find("attribute")
	if attr is not None:
		print("attributes:", attr.attrib)
		print("vehicle type:", attr.attrib.get("vehicle_type"))

	occlusion = first_target.find("occlusion")

	if occlusion is not None:
		regions = occlusion.findall("region_overlap")
		print("occlusion region count:", len(regions))

		for region in regions:
			print("occlusion:", region.attrib)
	else:
		print("occlusion: None")


def main():
	parser = argparse.ArgumentParser(
		description="Inspect UA-DETRAC XML annotation structure."
	)

	parser.add_argument(
		"--xml",
		required=True,
		help="Path to a UA-DETRAC XML annotation file.",
	)

	args = parser.parse_args()

	inspect_xml(args.xml)


if __name__ == "__main__":
	main()
