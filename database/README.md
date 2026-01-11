# Database Layer

This folder contains the PostgreSQL schema and migration scripts used by the platform.

The database stores:
- Users
- Processing jobs
- Uploaded photos
- Generated 3D models
- Hairstyle metadata
- Try-on history

Binary data (images, meshes) are never stored in the database — only secure storage URLs are saved.
