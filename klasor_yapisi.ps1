# 3D Hair Try-On Platform - Windows Setup Script
# Run this in PowerShell: .\setup-project-windows.ps1

Write-Host "🚀 Creating 3D Hair Try-On Platform structure..." -ForegroundColor Green

# Create main project directory
$projectName = "3d-hair-tryon"
New-Item -ItemType Directory -Force -Path $projectName | Out-Null
Set-Location $projectName

# Initialize Git
git init
Write-Host "✅ Git initialized" -ForegroundColor Cyan

# Main directories
$mainDirs = @(
    "mobile-app",
    "backend",
    "ai-pipeline",
    "hair-library",
    "database",
    "docker",
    "docs",
    "tests",
    "scripts"
)

foreach ($dir in $mainDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "✅ Main directories created" -ForegroundColor Cyan

# AI Pipeline subdirectories
$aiPipelineDirs = @(
    "ai-pipeline\01-photo-validation",
    "ai-pipeline\02-hair-segmentation",
    "ai-pipeline\02-hair-segmentation\models",
    "ai-pipeline\02-hair-segmentation\models\weights",
    "ai-pipeline\02-hair-segmentation\utils",
    "ai-pipeline\03-photogrammetry",
    "ai-pipeline\03-photogrammetry\configs",
    "ai-pipeline\04-mesh-cleanup",
    "ai-pipeline\05-hair-fitting",
    "ai-pipeline\06-quality-control",
    "ai-pipeline\common"
)

foreach ($dir in $aiPipelineDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "✅ AI Pipeline structure created" -ForegroundColor Cyan

# Backend subdirectories
$backendDirs = @(
    "backend\app",
    "backend\app\api",
    "backend\app\services",
    "backend\app\utils",
    "backend\workers",
    "backend\alembic"
)

foreach ($dir in $backendDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "✅ Backend structure created" -ForegroundColor Cyan

# Mobile App subdirectories
$mobileAppDirs = @(
    "mobile-app\src",
    "mobile-app\src\screens",
    "mobile-app\src\components",
    "mobile-app\src\services",
    "mobile-app\src\utils",
    "mobile-app\assets"
)

foreach ($dir in $mobileAppDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "✅ Mobile app structure created" -ForegroundColor Cyan

# Hair Library structure
$hairLibraryDirs = @(
    "hair-library\hair_001_short_male",
    "hair-library\hair_002_long_female",
    "hair-library\hair_003_curly_afro"
)

foreach ($dir in $hairLibraryDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "✅ Hair library structure created" -ForegroundColor Cyan

# Create essential files
@"
# 3D Hair Try-On Platform

Production-grade 3D hair try-on system using photogrammetry and AI.

## Quick Start

1. Install dependencies
2. Set up environment variables
3. Run Docker Compose
4. Start development

See `/docs` for detailed documentation.
"@ | Out-File -FilePath "README.md" -Encoding UTF8

@"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# AI Model Weights
ai-pipeline/02-hair-segmentation/models/weights/*.pth
ai-pipeline/02-hair-segmentation/models/weights/*.pt

# Data
*.obj
*.ply
*.pcd
data/
temp/
output/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Node
node_modules/
mobile-app/node_modules/

# Build
build/
dist/
*.log

# Docker
docker-compose.override.yml
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8

@"
# Environment Variables Template
# Copy this to .env and fill in your values

# Database
POSTGRES_USER=hairtryon_user
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=hairtryon_db
DATABASE_URL=postgresql://hairtryon_user:your_secure_password_here@localhost:5432/hairtryon_db

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage (MinIO for local, S3 for production)
STORAGE_TYPE=minio
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=hairtryon-photos
# AWS_ACCESS_KEY_ID=your_aws_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret
# AWS_BUCKET_NAME=hairtryon-prod

# JWT
JWT_SECRET_KEY=your_super_secret_jwt_key_change_this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Metashape
METASHAPE_LICENSE_KEY=your_metashape_license_key

# GPU
CUDA_VISIBLE_DEVICES=0

# API
API_HOST=0.0.0.0
API_PORT=8000
"@ | Out-File -FilePath ".env.example" -Encoding UTF8

Write-Host "✅ Essential files created" -ForegroundColor Cyan

# Create initial README files for key modules
@"
# Hair Segmentation Module

This module handles hair removal from photos using AI models.

## Models
- BiSeNet (Face Parsing)
- MODNet Hair (Hair Matting)

## Usage
\`\`\`python
python segment.py --input photos/ --output masks/
\`\`\`

## Output
- face_mask.png (only face + skull + ears)
- hair_mask.png (only hair region)
"@ | Out-File -FilePath "ai-pipeline\02-hair-segmentation\README.md" -Encoding UTF8

@"
# Photogrammetry Module

3D reconstruction using Metashape Pro.

## Requirements
- Metashape Pro license
- CUDA-capable GPU
- Masked photos (no hair)

## Process
1. Load masked photos
2. Align photos
3. Build dense point cloud
4. Generate mesh + UV
5. Export clean bald head model
"@ | Out-File -FilePath "ai-pipeline\03-photogrammetry\README.md" -Encoding UTF8

Write-Host ""
Write-Host "🎉 PROJECT STRUCTURE CREATED SUCCESSFULLY!" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Project location: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""
Write-Host "🚀 Next steps:" -ForegroundColor Cyan
Write-Host "  1. Copy .env.example to .env and configure"
Write-Host "  2. Install Python dependencies"
Write-Host "  3. Download AI model weights"
Write-Host "  4. Test hair segmentation module"
Write-Host ""
Write-Host "📖 Check README.md for documentation" -ForegroundColor Cyan