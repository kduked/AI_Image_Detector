import exifread


def extract_metadata(image_path):
    metadata = {}

    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
    
        metadata['has_exif'] = bool(tags)
        metadata['camera_model'] = str(tags.get('Image Model', 'Unknown'))
        metadata['software'] = str(tags.get('Image Software', 'Unknown'))
        metadata['datetime'] = str(tags.get('Exif DateTimeOriginal', 'Unknown'))

    except Exception as e:
        metadata['error'] = str(e)

    return metadata

def analyze_metadata(metadata):
    flags = []

    if not metadata.get('has_exif'):
        flags.append('No EXIF data found')

    if metadata.get('software') != 'Unknown':
        flags.append(f'Software tag present: {metadata['software']}')

    if metadata.get('camera_model') == 'Unknown':
        flags.append('Missing camera model')

    return flags

#test of main 

def main():
    # Hardcode your image path here
    image_path = "C:/Users/Dylan/.vscode/AI Image Detector/data/dog.jfif"

    # Extract metadata
    metadata = extract_metadata(image_path)

    if 'error' in metadata:
        print(f"Error reading metadata: {metadata['error']}")
        return

    print("\n=== Metadata Extracted ===")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Analyze metadata for suspicious/missing info
    flags = analyze_metadata(metadata)

    print("\n=== Metadata Analysis ===")
    if flags:
        for f in flags:
            print(f"{f}")
    else:
        print("No suspicious metadata flags found")

    # Simple AI likelihood calculator
    ai_score = 0
    if not metadata.get('has_exif'):
        ai_score += 1
    if metadata.get('camera_model') == 'Unknown':
        ai_score += 1
    if metadata.get('software') != 'Unknown':
        ai_score += 1

    # Decide likely AI
    if ai_score >= 2:  # 2 or more flags -> likely AI
        print("\nResult: Likely AI-generated image")
    else:
        print("\nResult: Likely natural image")


if __name__ == "__main__":
    main()