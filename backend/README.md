# Backend API

This folder contains the FastAPI backend responsible for:

- User authentication
- Photo upload and validation
- Job management and status tracking
- Storage integration (S3 / MinIO)
- Triggering AI pipeline jobs
- Returning 3D model URLs to the client

The backend communicates with GPU workers via Redis and Celery and stores metadata in PostgreSQL.
