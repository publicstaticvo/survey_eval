import chardet

def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        raw_data = f.read()
    
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    
    if encoding == 'utf-8':
        print(f"Detect codec: UTF-8. Confidence: {confidence:.2f}. Pass.")
    else:
        print(f"Detect codec: {encoding}. Confidence: {confidence:.2f}.")
    
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
        return content, encoding
    except:
        print(f"Not codec {encoding}, Try others")
        return "", None
