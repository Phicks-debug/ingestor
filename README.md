# Ingesting document service

## Overview

This is the ingestion service. Responsible for upload document, parsing, processing or chunking. Passing to embedding model to convert to sematic vector. Finally, save those semantic vector into desirable database.

The service is auto sync and self managable.
You can also manage sync tasks and document progress through documents/ api status.

## Run test

- run make qdrant-up to setup qdrant database locally first
- run `pytest tests.py -s -v ` to test with logs and verbose output

## Run development

- run `make run` to run the application in local for development

## Run deployment

- Start docker
- run `make docker-build` to bulild the application
- run `make docker-run` to run the application in docker container. Other command can be found in the Makefile

## API Endpoints

### Health Check

**GET** `/health`

Check if the service is running.

**Response:**
```json
{
  "status": "ok"
}
```

### Upload Documents

**POST** `/documents/upload`

Upload multiple PDF documents to S3 for processing.

**Request:** Multipart form data with file uploads

**Response:**
```json
{
  "messages": [
    {"filename": "document.pdf", "status": "uploaded"},
    {"filename": "doc2.pdf", "status": "failed", "error": "error message"}
  ]
}
```

### Trigger Document Sync

**POST** `/documents/sync`

Trigger asynchronous processing of documents from S3. Returns immediately with a task ID for status tracking.

**Response:**
```json
{
  "task_id": "uuid-string",
  "message": "Document sync started in background",
  "status_url": "/documents/sync/status/{task_id}"
}
```

### Get All Sync Task Status

**GET** `/documents/sync/status`

Get status of all background sync tasks.

**Response:**
```json
{
  "tasks": [
    {
      "task_id": "uuid-string",
      "status": "running",
      "started_at": "2025-10-17T10:30:00",
      "completed_at": null,
      "total_documents": 10,
      "processed_documents": 5,
      "successful_documents": 4,
      "skipped_documents": 1,
      "failed_documents": 0,
      "current_operation": "Processing document.pdf"
    }
  ]
}
```

### Get Specific Sync Task Status

**GET** `/documents/sync/status/{task_id}`

Get detailed status of a specific background sync task.

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "started_at": "2025-10-17T10:30:00",
  "completed_at": "2025-10-17T10:35:00",
  "total_documents": 10,
  "processed_documents": 10,
  "successful_documents": 8,
  "skipped_documents": 1,
  "failed_documents": 1,
  "failed_files": [
    {"filename": "error.pdf", "error": "timeout error"}
  ],
  "current_operation": "Completed successfully",
  "error_message": null
}
```

### Get Document Details for Sync Task

**GET** `/documents/sync/{task_id}/documents`

Get detailed status of all documents in a specific sync task. Shows which documents are pending, processing, done, or failed.

**Response:**
```json
{
  "task_id": "uuid-string",
  "documents": [
    {
      "filename": "document.pdf",
      "s3_key": "documents/document.pdf",
      "status": "success",
      "retry_count": 0,
      "error_message": null
    },
    {
      "filename": "failed.pdf",
      "s3_key": "documents/failed.pdf",
      "status": "failed",
      "retry_count": 3,
      "error_message": "Model timeout error"
    }
  ]
}
```

**Document Status Values:**
- `pending` - Document queued for processing
- `processing` - Document is being converted and processed
- `embedding` - Generating embeddings for document
- `ingesting` - Upserting vectors to database
- `success` - Document processed successfully
- `skipped` - Document unchanged, skipped processing
- `throttled` - Rate limited, will retry
- `failed` - Processing failed

### Delete Document

**DELETE** `/documents/{document_id}`

Delete a document by ID (placeholder endpoint).

**Response:**
```json
{
  "status": "deleted",
  "document_id": 123
}
```

## Limitation

- Model throttle and runtime. (Sometime model can be timeout)
- Currently only support PDF files. Will support variety of other files as well.