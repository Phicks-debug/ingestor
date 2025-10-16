import os
import time
import logsim

log = logsim.CustomLogger()


def test_parsing_function():
    from main import parsing

    sample_text = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 55 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Hello, World!) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000067 00000 n \n0000000123 00000 n \n0000000221 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n321\n%%EOF"  # noqa: E501
    start = time.time()
    parsed_text = parsing(sample_text)
    log.info("Process took %s seconds", time.time() - start)
    assert "Hello, World!" in parsed_text
    assert isinstance(parsed_text, str)
    assert len(parsed_text) > 0


def test_parsing_function_with_real_pdf():
    from main import parsing

    # Load PDF from test_samples directory
    pdf_path = "test_samples/test_docs.pdf"
    with open(pdf_path, "rb") as f:
        sample_pdf = f.read()

    start = time.time()
    parsed_text = parsing(sample_pdf)
    log.info("Process took %s seconds", time.time() - start)
    log.info(f"Parsed text length: {len(parsed_text)} characters")
    log.debug(f"Parsed text {parsed_text}")
    assert "valued at approximately USD 3.92 billion in 2024" in parsed_text
    assert isinstance(parsed_text, str)
    assert len(parsed_text) > 0


def test_chunking_function():
    from main import chunking

    text = "This is a test document. " * 10000
    start = time.time()
    chunks = chunking(text)

    log.info("Process took %s seconds", time.time() - start)
    log.info(f"Original text length: {len(text)} characters")
    log.info(f"Generated {len(chunks)} chunks.")

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 0


def test_embedding_texts_function():
    from main import embedding_texts

    chunks = ["This is a test chunk.", "Another test chunk."]
    start = time.time()
    result = embedding_texts(chunks)
    embeddings = result["embeddings"]["float"]
    texts = result["texts"]
    log.info("Process took %s seconds", time.time() - start)
    assert len(embeddings) == len(chunks)
    assert len(texts) == len(chunks)
    for emb in embeddings:
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)
    for text in texts:
        assert isinstance(text, str)


def test_embedding_images_function():
    import base64
    from main import embedding_images

    # Load image from test_samples and convert to base64
    image_path = "test_samples/test_image.png"

    _, file_extension = os.path.splitext(image_path)
    file_type = file_extension[1:]
    with open(image_path, "rb") as f:
        enc_img = base64.b64encode(f.read()).decode("utf-8")
        enc_img = f"data:image/{file_type};base64,{enc_img}"

    # Pass as list of base64 strings
    base64_images = [enc_img]

    start = time.time()
    result = embedding_images(base64_images)
    embeddings = result["embeddings"]["float"]
    images = result["images"]
    log.info("Process took %s seconds", time.time() - start)
    assert len(embeddings) == len(base64_images)
    assert len(images) == len(base64_images)
    for emb in embeddings:
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)
    for img in images:
        assert isinstance(img, dict)


def test_upsert_database_function():
    from main import upsert_database

    # Test data - simulating text chunks
    chunks = [
        "This is the first test chunk of text.",
        "This is the second test chunk of text.",
        "This is the third test chunk of text.",
    ]

    # Simulated Bedrock embedding response for text
    embeddings = {
        "embeddings": {
            "float": [
                [0.1] * 1024,  # 1024-dimensional vector
                [0.2] * 1024,
                [0.3] * 1024,
            ]
        },
        "test": chunks,
    }

    metadata = {
        "filename": "test_document.pdf",
        "s3_key": "documents/test_document.pdf",
    }

    # Call the function
    start = time.time()
    upsert_database(chunks, embeddings, metadata)
    log.info("Process took %s seconds", time.time() - start)


def test_sync_document_from_s3_function():
    from main import sync_document_from_s3

    # This test assumes that there are documents in the S3 bucket.
    # It will run the sync function and check for any exceptions.
    try:
        start = time.time()
        sync_document_from_s3()
        log.info("Process took %s seconds", time.time() - start)
    except Exception as e:
        assert False, f"sync_document_from_s3 raised an exception: {e}"
