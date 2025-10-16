import os
import time
import fitz
import json
import boto3
import base64
import orjson
import semchunk
import multiprocessing
import uuid
import hashlib
import asyncio
import aiobedrock

from logsim import CustomLogger
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from utils import load_tokenizer, render_template
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()
load_tokenizer()


AWS_REGION = os.getenv("AWS_REGION")

S3_DOCUMENTS_BUCKET = os.getenv("S3_DOCUMENTS_BUCKET")
S3_IMAGES_BUCKET = os.getenv("S3_IMAGES_BUCKET")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Document hash tracking file
DOCUMENT_HASH_FILE = os.path.join(
    os.path.dirname(__file__),
    ".document_hashes.json",
)

PARSING_MODEL_ID = os.getenv("PARSING_MODEL_ID")

CHUNK_TYPE = os.getenv("CHUNK_TYPE")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))

EMBEDDING_OUTPUT_TYPE = os.getenv("EMBEDDING_OUTPUT_TYPE")
EMBEDDING_OUTPUT_DIMENSION = int(os.getenv("EMBEDDING_OUTPUT_DIMENSION"))

_ASSUME_ROLE_MAP = {
    "s3": lambda: os.getenv("AWS_S3_ASSUME_ROLE_ARN"),
    "bedrock": lambda: os.getenv("AWS_BEDROCK_ASSUME_ROLE_ARN"),
}

app = FastAPI()
log = CustomLogger()


@app.get("/health")
def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "ok"},
    )


@app.get("/documents/status")
def get_documents_status():
    pass


@app.post("/documents/upload")
async def upload_document(files: list[UploadFile] = File(...)):
    """
    Upload multiple documents to S3 for processing.
    """
    s3_client = __create_aws_client("s3")
    results = []

    for file in files:
        # Read file content
        content = await file.read()

        # Upload to S3
        try:
            s3_key = f"documents/{file.filename}"
            s3_client.put_object(
                Bucket=S3_DOCUMENTS_BUCKET,
                Key=s3_key,
                Body=content,
                ContentType=file.content_type or "application/octet-stream",
            )
            results.append({"filename": file.filename, "status": "uploaded"})
        except Exception as e:
            results.append(
                {
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e),
                }
            )

    return JSONResponse(
        status_code=200,
        content={"messages": results},
    )


@app.post("/documents/sync")
async def sync_document(background_tasks: BackgroundTasks):
    """
    Trigger async processing of documents from S3.
    Returns immediately while processing happens in the background.
    """
    background_tasks.add_task(sync_document_from_s3)

    return JSONResponse(
        status_code=202,
        content={"messages": "Document sync started in background"},
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: int):
    # Simulate document deletion
    return {"status": "deleted", "document_id": document_id}


def __load_document_hashes() -> dict:
    """
    Load document hashes from the tracking file.
    Returns a dict mapping s3_key -> hash.
    """
    if os.path.exists(DOCUMENT_HASH_FILE):
        try:
            with open(DOCUMENT_HASH_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load document hashes: {e}")
            return {}
    return {}


def __save_document_hashes(hashes: dict):
    """
    Save document hashes to the tracking file.
    """
    try:
        with open(DOCUMENT_HASH_FILE, "w") as f:
            json.dump(hashes, f, indent=2)
    except Exception as e:
        log.exception(f"Failed to save document hashes: {e}")


def __calculate_content_hash(content: bytes) -> str:
    """
    Calculate SHA256 hash of document content.
    """
    return hashlib.sha256(content).hexdigest()


def __delete_document_from_qdrant(s3_key: str):
    """
    Delete all vectors associated with a document from Qdrant.
    """
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            https=False,
        )

        # Delete all points with matching s3_key
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="s3_key",
                        match=MatchValue(value=s3_key),
                    )
                ]
            ),
        )
        log.info(f"Deleted vectors for document: {s3_key}")
    except Exception as e:
        log.exception(f"Failed to delete document from Qdrant: {e}")


def sync_document_from_s3():
    """
    Background task to process documents from S3.
    Tracks document hashes to only process new/modified documents.
    Automatically deletes vectors for removed documents.
    """
    s3_client = __create_aws_client("s3")

    try:
        # Load existing document hashes
        stored_hashes = __load_document_hashes()
        log.info(f"Loaded {len(stored_hashes)} document hashes")

        # List all documents in the S3 bucket
        response = s3_client.list_objects_v2(
            Bucket=S3_DOCUMENTS_BUCKET,
            Prefix="documents/",
        )

        if "Contents" not in response:
            log.info("No documents found in S3")
            # If no documents in S3 but we have stored hashes, delete all
            if stored_hashes:
                log.info("Deleting all documents from Qdrant")
                for s3_key in stored_hashes.keys():
                    __delete_document_from_qdrant(s3_key)
                stored_hashes.clear()
                __save_document_hashes(stored_hashes)
            return

        # Track current S3 documents
        current_s3_keys = set()
        updated_hashes = {}

        # Process each document
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            filename = s3_key.split("/")[-1]

            # Skip if it's just the folder itself
            if not filename:
                continue

            current_s3_keys.add(s3_key)

            try:
                # Download document from S3
                file_obj = s3_client.get_object(
                    Bucket=S3_DOCUMENTS_BUCKET,
                    Key=s3_key,
                )
                content = file_obj["Body"].read()

                # Calculate content hash
                content_hash = __calculate_content_hash(content)

                # Check if document has changed
                if (
                    s3_key in stored_hashes
                    and stored_hashes[s3_key] == content_hash  # noqa: E501
                ):
                    log.info(f"Skipping unchanged document: {filename}")
                    updated_hashes[s3_key] = content_hash
                    continue

                # Document is new or modified
                if s3_key in stored_hashes:
                    log.info(f"Document modified, reprocessing: {filename}")
                    # Delete old vectors before processing new version
                    __delete_document_from_qdrant(s3_key)
                else:
                    log.info(f"New document detected: {filename}")

                # # Decode content to text
                # log.info(f"Parsing document {filename} from S3")
                # text_content = parsing(content)
                # log.debug(f"Parsed text content: {text_content[:100]}...")
                # log.info(f"Parsed document {filename}")

                # # Chunk the document
                # log.info(f"Chunking document {filename}")
                # chunks = chunking(text_content)
                # log.debug(f"Chunks: {chunks}")
                # for i, chunk in enumerate(chunks):
                #     log.info(f"Chunk {i+1}: {len(chunk)} characters")
                #     log.debug(f"Ck {i+1} preview first 100c: {chunk[:100]}.")
                #     log.debug(f"Ck {i+1} preview last 100c: .{chunk[-100:]}")
                # log.info(f"Generated {len(chunks)} chunks for {filename}")

                # # Generate embeddings
                # log.info(msg=f"Generating embeddings for {filename}")
                # result = embedding_texts(chunks)
                # text_embeddings = result["embeddings"][EMBEDDING_OUTPUT_TYPE]
                # log.debug(f"Embeddings: {results}")
                # log.info(msg=f"Generated embeddings for {filename}")

                # # Upsert vectors in Qdrant vector database with metadata
                # metadata = {"filename": filename, "s3_key": s3_key}
                # asyncio.run(
                #     upsert_database(
                #         chunks=chunks,
                #         embeddings=text_embeddings,
                #         metadata=metadata,
                #     )
                # )

                # Convert PDF to images and generate S3 keys
                log.info(f"Converting PDF to images for {filename}")
                images, s3_keys, image_bytes_list = __convert_pdf_to_image(
                    file_content=content,
                    filename=filename,
                )
                log.info(f"Converted {len(images)} pages to images")

                # Upload images to S3 synchronously
                log.info(f"Uploading {len(image_bytes_list)} images to S3")
                __upload_images_to_s3(image_bytes_list, s3_keys, filename)
                log.info("S3 upload completed")

                # Generate embeddings
                log.info(msg=f"Generating embeddings for {filename}")
                image_embeddings = embedding_images(images)
                log.info(msg=f"Generated embeddings for {filename}")

                # Upsert vectors in Qdrant with S3 keys
                metadata = {"filename": filename, "s3_key": s3_key}
                upsert_database(
                    chunks=images,
                    is_image=True,
                    embedding_results=image_embeddings,
                    metadata=metadata,
                    s3_image_keys=s3_keys,
                )

                # Save hash after successful processing
                updated_hashes[s3_key] = content_hash
                log.info(f"Successfully processed {filename}")

            except Exception as e:
                log.exception(f"Failed to process {filename}: {str(e)}")
                # Don't update hash if processing failed
                continue

        # Detect and delete documents that were removed from S3
        deleted_keys = set(stored_hashes.keys()) - current_s3_keys
        if deleted_keys:
            log.info(f"Detected {len(deleted_keys)} deleted documents")
            for deleted_key in deleted_keys:
                deleted_filename = deleted_key.split("/")[-1]
                log.info(f"Deleting document from Qdrant: {deleted_filename}")
                __delete_document_from_qdrant(deleted_key)

        # Save updated hashes
        __save_document_hashes(updated_hashes)
        log.info(
            msg="Document sync completed."
            f"Processed {len(updated_hashes)} documents"  # noqa: E501
        )

    except Exception as e:
        log.exception(f"Error listing S3 documents: {str(e)}")


def parsing(file_content: bytes) -> str:
    """
    Parse document content based.
    """
    data = base64.b64encode(file_content).decode("utf-8")
    client = __create_aws_client("bedrock-runtime")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a document to extract text from.",
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": data,
                        },
                        "title": "Document to extract text from",
                    },
                ],
            },
        ],
        "system": [
            {
                "type": "text",
                "text": render_template("parsing_system.j2"),
                "cache_control": {
                    "type": "ephemeral",
                },
            }
        ],
    }
    start_time = time.time()
    stream = client.invoke_model_with_response_stream(
        modelId=PARSING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=orjson.dumps(body),
        trace="ENABLED",
    )
    response = []
    for e in stream["body"]:
        # Record the time taken to process the response
        if start_time:
            time_taken = time.time() - start_time
            log.info(f"TTFT for parsing: {time_taken:.2f} seconds")
            start_time = None
        chunk = json.loads(e["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            if chunk["delta"]["type"] == "text_delta":
                response.append(chunk["delta"]["text"])
                print(".", end="", flush=True)
    # Validate and clean up response
    response = __validate_parsing_response("".join(response))
    return response


def chunking(text: str) -> list[str]:
    """
    Chunk document into smaller pieces with overlap.
    Uses o200k_base tokenizer (200k vocab, 128k context).
    """
    # Use gpt-4o which uses o200k_base encoding (200k vocab size)
    chunker = semchunk.chunkerify("o200k_base", CHUNK_SIZE)
    # Define number of processes for parallel chunking
    # Use max-1 cores to leave one core free for system operations
    cpu_count = multiprocessing.cpu_count()
    process_num = max(1, cpu_count - 1)
    log.debug(f"Num processes for chunking: {process_num}")
    return chunker(text, processes=process_num)


def embedding_texts(chunks: list[str]) -> list:
    """
    Invoke Bedrock embedding model to get text embeddings.
    """
    client = __create_aws_client("bedrock-runtime")
    body = {
        "texts": chunks,
        "input_type": "search_document",
        "embedding_types": [EMBEDDING_OUTPUT_TYPE],
        "output_dimension": EMBEDDING_OUTPUT_DIMENSION,
        "max_tokens": 128000,
        "truncate": "NONE",
    }
    response = client.invoke_model(
        modelId="global.cohere.embed-v4:0",
        contentType="application/json",
        accept="application/json",
        body=orjson.dumps(body),
        trace="ENABLED",
    )
    return orjson.loads(response["body"].read())


async def __process_image_batch_async(
    client: aiobedrock.Client,
    batch_images: list[str],
    batch_num: int,
    total_batches: int,
    batch_start: int,
    batch_end: int,
) -> list:
    """
    Process a single batch of images asynchronously.
    """
    log.info(
        f"Processing batch {batch_num}/{total_batches} "
        f"(images {batch_start + 1}-{batch_end})"
    )

    body = {
        "images": batch_images,
        "input_type": "search_document",
        "embedding_types": [EMBEDDING_OUTPUT_TYPE],
        "output_dimension": EMBEDDING_OUTPUT_DIMENSION,
        "max_tokens": 128000,
        "truncate": "NONE",
    }

    try:
        response = await client.invoke_model(
            body=orjson.dumps(body),
            modelId="global.cohere.embed-v4:0",
            contentType="application/json",
            accept="application/json",
            trace="ENABLED",
        )

        batch_result = orjson.loads(response)
        batch_embeddings = batch_result["embeddings"][EMBEDDING_OUTPUT_TYPE]

        log.info(
            f"Successfully processed batch {batch_num}/{total_batches} "
            f"({len(batch_embeddings)} embeddings)"
        )

        return batch_embeddings

    except Exception as e:
        log.exception(
            msg="Failed to process batch"
            f" {batch_num}/{total_batches}: {str(e)}"  # noqa: E501
        )
        raise


async def __embedding_images_async(
    images: list[str],
    batch_size: int = 10,
) -> dict:
    """
    Async function to invoke Bedrock embedding model for images.
    Processes all batches concurrently using asyncio.
    """
    total_images = len(images)
    total_batches = (total_images + batch_size - 1) // batch_size

    log.info(
        f"Processing {total_images} images in "
        f"{total_batches} batches of {batch_size}"
    )

    # Create aiobedrock client with assume_role_arn
    async with aiobedrock.Client(
        region_name=AWS_REGION,
        assume_role_arn=__get_assume_role_arn("bedrock"),
    ) as client:
        # Create tasks for all batches
        tasks = []
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_images = images[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1

            task = __process_image_batch_async(
                client=client,
                batch_images=batch_images,
                batch_num=batch_num,
                total_batches=total_batches,
                batch_start=batch_start,
                batch_end=batch_end,
            )
            tasks.append(task)

        # Execute all batches concurrently
        all_batch_results = await asyncio.gather(*tasks)

        # Flatten all embeddings into a single list
        all_embeddings = []
        for batch_embeddings in all_batch_results:
            all_embeddings.extend(batch_embeddings)

    # Return result in the same format as the original function
    return {"embeddings": {EMBEDDING_OUTPUT_TYPE: all_embeddings}}


def embedding_images(images: list[str], batch_size: int = 10) -> dict:
    """
    Invoke Bedrock embedding model to get images embeddings.
    Processes images in batches concurrently
    using asyncio to avoid input length limits.
    """
    return asyncio.run(__embedding_images_async(images, batch_size))


def upsert_database(
    chunks: list[str],
    embedding_results: dict,
    metadata: dict,
    s3_image_keys: list[str] = None,
    is_image: bool = False,
):
    """
    Upsert vectors into Qdrant vector database.
    """
    # Create client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        https=False,
    )

    # Ensure collection exists
    try:
        client.get_collection(collection_name=QDRANT_COLLECTION)
        log.info(f"Collection '{QDRANT_COLLECTION}' already exists")
    except Exception:
        # Create collection if it doesn't exist
        log.info(f"Creating collection '{QDRANT_COLLECTION}'")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_OUTPUT_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

    embeddings = embedding_results["embeddings"][EMBEDDING_OUTPUT_TYPE]

    # Prepare points for upsert
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        # Generate a unique ID based on filename and chunk index
        point_id = hash(f"{metadata['filename']}_{idx}")

        # Create payload with metadata
        payload = {
            **metadata,
            "chunk_index": idx,
            "content_type": "image" if is_image else "text",
        }

        # For images, store S3 key instead of base64 data
        if is_image and s3_image_keys:
            payload["image_s3_key"] = s3_image_keys[idx]
        else:
            # For text, store the actual content
            payload["chunk_content"] = chunk

        # Create point
        point = PointStruct(
            id=abs(point_id),  # Qdrant requires positive integer IDs
            vector=vector,
            payload=payload,
        )
        points.append(point)

    # Upsert points to Qdrant
    log.info(f"Upserting {len(points)} points")
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )
    log.info(
        msg=f"Successfully upserted {len(points)} "
        f"vectors for {metadata['filename']}"
    )


def __validate_parsing_response(response: str) -> str:
    """
    Validate and clean up parsing response.
    """
    if not response.startswith("<markdown>") or not response.endswith(
        "</markdown>"
    ):  # noqa: E501
        raise ValueError("Parsed text has malformed markdown tags")
    if response.startswith("<markdown>"):
        response = response[10:]
    if response.endswith("</markdown>"):
        response = response[:-11]
    if not response.strip():
        raise ValueError("Parsed text is empty")
    return response


def __get_assume_role_arn(service_name: str) -> str | None:
    """
    Get the assume role ARN for a given service.
    """
    for service_key, role_getter in _ASSUME_ROLE_MAP.items():
        if service_key in service_name:
            role_arn = role_getter()
            return role_arn if role_arn else None
    return None


def __create_aws_session_with_assumed_role(
    service_name: str, role_arn: str
) -> boto3.Session:
    """
    Create a boto3 session using assumed role credentials.
    """
    sts_client = boto3.client("sts")
    response = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=f"{service_name}-session",
    )

    credentials = response["Credentials"]
    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=AWS_REGION,
    )


def __create_aws_client(service_name: str) -> boto3.client:
    """
    Create an AWS client, optionally assuming a role if configured.
    """
    role_arn = __get_assume_role_arn(service_name)

    if role_arn:
        # Create client with assumed role credentials
        session = __create_aws_session_with_assumed_role(
            service_name=service_name,
            role_arn=role_arn,
        )
        return session.client(
            service_name=service_name,
            region_name=AWS_REGION,
        )
    else:
        # Create client with default credentials
        return boto3.client(
            service_name=service_name,
            region_name=AWS_REGION,
        )


def __upload_images_to_s3(
    image_bytes_list: list[bytes],
    s3_keys: list[str],
    filename: str,
):
    """
    Upload images to S3 synchronously.
    """
    s3_client = __create_aws_client("s3")

    for idx, (image_bytes, s3_key) in enumerate(
        zip(image_bytes_list, s3_keys),
    ):
        s3_client.put_object(
            Bucket=S3_IMAGES_BUCKET,
            Key=s3_key,
            Body=image_bytes,
            ContentType="image/png",
            Metadata={
                "source_document": filename,
                "page_number": str(idx),
            },
        )
        log.debug(f"Uploaded image to S3: {s3_key}")

    log.info(f"Successfully uploaded {len(image_bytes_list)} images to S3")


def __convert_pdf_to_image(
    file_content: bytes,
    filename: str,
) -> tuple[list[str], list[str], list[bytes]]:
    """
    Convert PDF file to a list of base64-encoded PNG images as data URLs.
    Also generates S3 keys for each image upfront.
    Only one image per page.
    Returns: tuple of (base64_images, s3_keys, image_bytes_list)
    Processes each page individually to minimize memory usage.
    """
    base64_images = []
    s3_keys = []
    image_bytes_list = []

    # Open PDF from bytes
    pdf_document = fitz.open(stream=file_content, filetype="pdf")

    try:
        # Process each page individually
        for page_num in range(pdf_document.page_count):
            # Load one page at a time
            page = pdf_document.load_page(page_num)

            # Render page to pixmap (image) at 300 DPI for good quality
            matrix = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=matrix)

            # Convert pixmap to PNG bytes
            png_bytes = pix.pil_tobytes(format="PNG")

            # Encode to base64 and format as data URL
            image_base64 = base64.b64encode(png_bytes).decode("utf-8")
            image_data_url = f"data:image/png;base64,{image_base64}"
            base64_images.append(image_data_url)

            # Generate S3 key for this image
            doc_name = os.path.splitext(filename)[0]
            s3_key = f"images/{doc_name}/page_{page_num}_{uuid.uuid4()}.png"
            s3_keys.append(s3_key)

            # Store the raw bytes for later upload
            image_bytes_list.append(png_bytes)

            # Clean up page resources
            pix = None
            page = None

    finally:
        # Close the PDF document
        pdf_document.close()

    return base64_images, s3_keys, image_bytes_list
