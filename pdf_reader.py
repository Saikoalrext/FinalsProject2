import fitz

def extract_pdf_text(pdf_path):
    try:
        doc= fitz.open(pdf_path)
        text= ""
        for page in doc:
            text+= page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""