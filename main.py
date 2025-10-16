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

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from logsim import CustomLogger
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from utils import load_tokenizer, load_hashes, render_template
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
load_hashes()
load_tokenizer()


AWS_REGION = os.getenv("AWS_REGION")

S3_DOCUMENTS_BUCKET = os.getenv("S3_DOCUMENTS_BUCKET")
S3_IMAGES_BUCKET = os.getenv("S3_IMAGES_BUCKET")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

PARSING_MODEL_ID = os.getenv("PARSING_MODEL_ID")
DOCUMENT_HASH_FILE = os.getenv("DOCUMENT_HASH_FILE")

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


class TaskStatus(Enum):
    """Status of document processing task"""

    PENDING = "pending"
    SUCCESS = "success"
    THROTTLED = "throttled"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DocumentTask:
    """Represents a document processing task"""

    s3_key: str
    filename: str
    content_hash: str
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None


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


def sync_document_from_s3():
    """
    Background task to process documents from S3 with retry mechanism.
    Tracks document hashes to only process new/modified documents.
    Handles throttling with up to 3 retry attempts and 60-second delays.
    """
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 60

    s3_client = __create_aws_client("s3")

    try:
        # Load existing document hashes
        stored_hashes = __load_document_hashes()
        log.info(f"Loaded {len(stored_hashes)} document hashes\n")

        # List all documents in the S3 bucket
        response = s3_client.list_objects_v2(
            Bucket=S3_DOCUMENTS_BUCKET,
            Prefix="documents/",
        )

        if "Contents" not in response:
            log.info("No documents found in S3")
            if stored_hashes:
                log.info("Deleting all documents from Qdrant")
                for s3_key in stored_hashes.keys():
                    __delete_document_from_qdrant(s3_key)
                stored_hashes.clear()
                __save_document_hashes(stored_hashes)
            return

        # Create task list for all documents
        tasks: list[DocumentTask] = []
        current_s3_keys = set()

        log.info("=" * 60)
        log.info("PREPARING DOCUMENT PROCESSING TASKS")
        log.info("=" * 60)

        for obj in response["Contents"]:
            s3_key = obj["Key"]
            filename = s3_key.split("/")[-1]

            if not filename:
                continue

            current_s3_keys.add(s3_key)

            # Download just to get the content hash
            try:
                file_obj = s3_client.get_object(
                    Bucket=S3_DOCUMENTS_BUCKET,
                    Key=s3_key,
                )
                content = file_obj["Body"].read()
                content_hash = __calculate_content_hash(content)

                task = DocumentTask(
                    s3_key=s3_key,
                    filename=filename,
                    content_hash=content_hash,
                )
                tasks.append(task)
                log.info(f"+ Added task for: {filename}")
            except Exception as e:
                log.exception(f"Failed to create task for {filename}: {e}")
                continue

        log.info(f"** Total tasks created: {len(tasks)}\n")

        # Process documents with retry logic
        updated_hashes = {}
        retry_round = 0

        while retry_round <= MAX_RETRIES:
            # Get tasks that need processing
            pending_tasks = [
                t
                for t in tasks
                if t.status in [TaskStatus.PENDING, TaskStatus.THROTTLED]
            ]

            if not pending_tasks:
                break

            if retry_round > 0:
                log.info("=" * 60)
                log.info(f"RETRY ROUND {retry_round}/{MAX_RETRIES}")
                log.info("=" * 60)
                log.info(
                    f"Waiting {RETRY_DELAY_SECONDS} seconds "
                    "to refresh throttle limits..."
                )
                time.sleep(RETRY_DELAY_SECONDS)

            log.info(f"<< Processing {len(pending_tasks)} documents...\n")

            for task in pending_tasks:
                if task.status == TaskStatus.THROTTLED:
                    task.retry_count += 1
                    log.info(
                        f"<< Retrying {task.filename} "
                        f"(attempt {task.retry_count}/{MAX_RETRIES})"
                    )

                try:
                    success = __process_single_document(
                        task,
                        s3_client,
                        stored_hashes,
                    )

                    if success:
                        # Document processed successfully or skipped
                        if task.status == TaskStatus.SUCCESS:
                            updated_hashes[task.s3_key] = task.content_hash
                        elif task.status == TaskStatus.SKIPPED:
                            updated_hashes[task.s3_key] = task.content_hash

                except Exception as e:
                    log.exception(f"Failed to process {task.filename}: {e}")
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)

            retry_round += 1

        # Mark remaining throttled tasks as failed after max retries
        for task in tasks:
            if task.status == TaskStatus.THROTTLED:
                task.status = TaskStatus.FAILED
                log.error(
                    f"{task.filename} failed after {MAX_RETRIES} "
                    f"retry attempts: {task.error_message}\n"
                )

        # Print summary
        log.info("=" * 60)
        log.info("PROCESSING SUMMARY")
        log.info("=" * 60)

        success_count = sum(1 for t in tasks if t.status == TaskStatus.SUCCESS)
        skipped_count = sum(1 for t in tasks if t.status == TaskStatus.SKIPPED)
        failed_count = sum(1 for t in tasks if t.status == TaskStatus.FAILED)

        log.info(f"✓ Successful: {success_count}")
        log.info(f"⊘ Skipped:    {skipped_count}")
        log.info(f"✗ Failed:     {failed_count}")
        log.info(f"= Total:      {len(tasks)}")

        if failed_count > 0:
            log.info("FAILED DOCUMENTS:")
            for task in tasks:
                if task.status == TaskStatus.FAILED:
                    log.error(f"  - {task.filename}: {task.error_message}")

        log.info("=" * 60 + "\n")

        # Detect and delete documents removed from S3
        deleted_keys = set(stored_hashes.keys()) - current_s3_keys
        if deleted_keys:
            log.info(f"Detected {len(deleted_keys)} deleted documents")
            for deleted_key in deleted_keys:
                deleted_filename = deleted_key.split("/")[-1]
                log.info(f"Deleting document from Qdrant: {deleted_filename}")
                __delete_document_from_qdrant(deleted_key)

        # Save updated hashes
        __save_document_hashes(updated_hashes)
        log.info("Document sync completed")
        log.debug(f"Processed {len(updated_hashes)} documents")

    except Exception as e:
        log.exception(f"Error in document sync: {str(e)}")


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


def __check_cached_images_in_s3(filename: str) -> tuple[bool, list[str]]:
    """
    Check if images for a document already exist in S3.
    Returns: (exists: bool, s3_keys: list[str])
    """
    s3_client = __create_aws_client("s3")
    doc_name = os.path.splitext(filename)[0]
    prefix = f"images/{doc_name}/"

    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_IMAGES_BUCKET,
            Prefix=prefix,
        )

        if "Contents" not in response or len(response["Contents"]) == 0:
            return False, []

        # Extract and sort S3 keys by page number
        s3_keys = []
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            # Skip directory markers
            if s3_key.endswith("/"):
                continue
            s3_keys.append(s3_key)

        # Sort by page number extracted from filename (page_X_uuid.png)
        def extract_page_num(key):
            try:
                basename = key.split("/")[-1]  # Get filename
                page_part = basename.split("_")[1]  # Get page number
                return int(page_part)
            except (IndexError, ValueError):
                return 0

        s3_keys.sort(key=extract_page_num)

        log.info(f"Found {len(s3_keys)} cached images in S3")
        return True, s3_keys

    except Exception as e:
        log.warning(f"Error checking cached images: {e}")
        return False, []


def __download_cached_images_from_s3(s3_keys: list[str]) -> list[str]:
    """
    Download cached images from S3 and convert to base64 data URLs.
    Returns: list of base64-encoded data URLs
    """
    s3_client = __create_aws_client("s3")
    base64_images = []

    for s3_key in s3_keys:
        try:
            response = s3_client.get_object(
                Bucket=S3_IMAGES_BUCKET,
                Key=s3_key,
            )
            image_bytes = response["Body"].read()

            # Convert to base64 data URL
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:image/png;base64,{image_base64}"
            base64_images.append(image_data_url)

        except Exception as e:
            log.error(f"Failed to download cached image {s3_key}: {e}")
            raise

    log.info(f"Downloaded {len(base64_images)} cached images from S3")
    return base64_images


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


def __is_timeout_error(exception: Exception) -> bool:
    """
    Check if an exception is a timeout error (HTTP 408).
    These errors should fail immediately without retry.
    """
    error_str = str(exception).lower()
    exception_type = type(exception).__name__.lower()

    # Check for timeout error indicators
    return (
        "408" in error_str
        or "modeltimeoutexception" in error_str
        or "modeltimeoutexception" in exception_type
        or ("timeout" in error_str and "model" in error_str)
    )


def __is_throttle_error(exception: Exception) -> bool:
    """
    Check if an exception is a throttle/rate limit error (HTTP 429).
    """
    error_str = str(exception).lower()
    exception_type = type(exception).__name__.lower()

    # Check for common throttle error indicators
    # Note: Explicitly exclude 408 timeout errors
    return (
        "429" in error_str
        or "throttl" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
        or "too many tokens" in error_str
        or "server disconnected" in error_str
        or "serverdisconnectederror" in exception_type
    ) and not __is_timeout_error(exception)


def __process_single_document(
    task: DocumentTask,
    s3_client,
    stored_hashes: dict,
) -> bool:
    """
    Process a single document task.
    Returns True if successful, False if throttled (needs retry).
    Raises exception for other errors.
    """
    try:
        # Download document from S3
        file_obj = s3_client.get_object(
            Bucket=S3_DOCUMENTS_BUCKET,
            Key=task.s3_key,
        )
        content = file_obj["Body"].read()

        # Check if document has changed
        if (
            task.s3_key in stored_hashes
            and stored_hashes[task.s3_key] == task.content_hash
        ):
            log.info(f"Skipping unchanged document: {task.filename}")
            task.status = TaskStatus.SKIPPED
            return True

        # Document is new or modified
        document_changed = task.s3_key in stored_hashes
        if document_changed:
            log.info(f"Document modified, reprocessing: {task.filename}")
            __delete_document_from_qdrant(task.s3_key)
        else:
            log.info(f"New document detected: {task.filename}")

        # Check if we can reuse cached images from S3
        images = None
        s3_keys = None
        use_cached_images = False

        # Only use cached images if document hasn't changed
        if not document_changed:
            c_exists, c_s3_keys = __check_cached_images_in_s3(task.filename)
            if c_exists:
                log.info(f"Using cached images from S3 for {task.filename}")
                try:
                    images = __download_cached_images_from_s3(c_s3_keys)
                    s3_keys = c_s3_keys
                    use_cached_images = True
                except Exception as e:
                    log.warning(
                        f"Failed to download cached images: {e}. "
                        "Will convert PDF instead."
                    )
                    use_cached_images = False

        # Convert PDF to images if cache not available or download failed
        if not use_cached_images:
            log.info(f"Converting PDF to images for {task.filename}")
            images, s3_keys, image_bytes_list = __convert_pdf_to_image(
                file_content=content,
                filename=task.filename,
            )
            log.info(f"Converted {len(images)} pages to images")

            # Upload images to S3
            log.info(f"Uploading {len(image_bytes_list)} images to S3")
            __upload_images_to_s3(image_bytes_list, s3_keys, task.filename)
            log.info("S3 upload completed")

        # Generate embeddings - this is where throttling might occur
        log.info(msg=f"Generating embeddings for {task.filename}")
        image_embeddings = embedding_images(images)
        log.info(msg=f"Generated embeddings for {task.filename}")

        # Upsert vectors in Qdrant with S3 keys
        metadata = {"filename": task.filename, "s3_key": task.s3_key}
        upsert_database(
            chunks=images,
            is_image=True,
            embedding_results=image_embeddings,
            metadata=metadata,
            s3_image_keys=s3_keys,
        )

        # Mark as successful
        task.status = TaskStatus.SUCCESS
        log.info(f"Successfully processed {task.filename}\n")
        return True

    except Exception as e:
        if __is_timeout_error(e):
            # Timeout error - mark as failed immediately without retry
            task.status = TaskStatus.FAILED
            error_summary = str(e).split("\n")[0][:150]
            task.error_message = error_summary
            log.error(
                f"FAILED: {task.filename} - "
                "Model timeout error. Task marked as failed.\n"
            )
            raise
        elif __is_throttle_error(e):
            # Throttle error - mark for retry
            task.status = TaskStatus.THROTTLED
            error_summary = str(e).split("\n")[0][:150]
            task.error_message = error_summary
            log.warning(
                f"THROTTLED: {task.filename} - "
                "Rate limit exceeded. Will retry after cooldown period.\n"
            )
            return False
        else:
            # Other error - re-raise
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            raise


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
        # Check error type and log appropriately
        if __is_timeout_error(e):
            # Timeout errors - log as error without full trace
            log.error(
                f"Batch {batch_num}/{total_batches} timed out: "
                f"{str(e).split(':')[0] if ':' in str(e) else str(e)[:100]}"
            )
        elif __is_throttle_error(e):
            # Throttle errors - log as warning
            log.warning(
                f"Batch {batch_num}/{total_batches} throttled/disconnected: "
                f"{str(e).split(':')[0] if ':' in str(e) else str(e)[:100]}"
            )
        else:
            # For other errors, log with full stack trace
            log.exception(
                "Failed to process batch"
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
