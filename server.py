"""
FastAPI server for TripoSR 3D model generation on RunPod.
Receives an image, generates a 3D model using TripoSR, converts to GLB,
and uploads to Supabase Storage.

Deploy on RunPod with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

# =====================================================
# Configuration
# =====================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Service role key for storage
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "models-3d")

# Path to TripoSR installation
TRIPOSR_PATH = os.getenv("TRIPOSR_PATH", "/root/TripoSR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/outputs")

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================
# FastAPI App
# =====================================================

app = FastAPI(
    title="TripoSR API",
    description="Generate 3D models from images using TripoSR",
    version="1.0.0",
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"📥 [FASTAPI] {request.method} {request.url.path}")
    print(f"📥 [FASTAPI] Headers: {dict(request.headers)}")
    if request.method == "POST":
        try:
            body = await request.body()
            print(f"📥 [FASTAPI] Body size: {len(body)} bytes")
            if len(body) < 500:
                print(f"📥 [FASTAPI] Body: {body.decode()[:500]}")
        except:
            pass
    response = await call_next(request)
    print(f"📤 [FASTAPI] Response: {response.status_code}")
    return response

# =====================================================
# Response Models
# =====================================================

class GenerateResponse(BaseModel):
    success: bool
    glb_url: Optional[str] = None
    error: Optional[str] = None
    task_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    triposr_available: bool
    gpu_available: bool
    gpu_info: Optional[str] = None
    error: Optional[str] = None

class ImageUrlRequest(BaseModel):
    image_url: str

# =====================================================
# Helper Functions
# =====================================================

async def run_triposr(image_path: str, output_dir: str) -> tuple[bool, str]:
    """
    Run TripoSR inference on the image.
    Returns (success, output_path or error_message)
    """
    try:
        # Build command
        cmd = [
            "python", "run.py",
            image_path,
            "--output-dir", output_dir,
            "--model-save-format", "obj",
            "--render",
        ]
        
        # Run TripoSR
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=TRIPOSR_PATH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            print(f"❌ TripoSR error: {error_msg}")
            
            # Check for missing module errors
            if "ModuleNotFoundError" in error_msg or "No module named" in error_msg:
                missing_module = error_msg.split("No module named")[-1].strip().replace("'", "").replace('"', '')
                return False, f"Missing Python module: {missing_module}. Install it with: pip install {missing_module}"
            
            return False, f"TripoSR failed: {error_msg}"
        
        # Find the output OBJ file
        obj_files = list(Path(output_dir).glob("**/*.obj"))
        if not obj_files:
            return False, "No OBJ file generated"
        
        return True, str(obj_files[0])
        
    except Exception as e:
        print(f"❌ Exception running TripoSR: {e}")
        return False, str(e)


async def convert_obj_to_glb(obj_path: str) -> tuple[bool, str]:
    """
    Convert OBJ to GLB using trimesh.
    Returns (success, glb_path or error_message)
    """
    try:
        import trimesh
        
        # Load the OBJ mesh
        mesh = trimesh.load(obj_path)
        
        # Generate GLB path
        glb_path = obj_path.replace(".obj", ".glb")
        
        # Export to GLB
        mesh.export(glb_path, file_type="glb")
        
        if os.path.exists(glb_path):
            return True, glb_path
        else:
            return False, "GLB file not created"
            
    except Exception as e:
        print(f"❌ Exception converting to GLB: {e}")
        return False, str(e)


async def upload_to_supabase(file_path: str, filename: str) -> tuple[bool, str]:
    """
    Upload file to Supabase Storage.
    Returns (success, public_url or error_message)
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        # If Supabase not configured, return local path
        return True, f"file://{file_path}"
    
    try:
        async with httpx.AsyncClient() as client:
            # Read file
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            # Upload to Supabase Storage
            upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
            
            response = await client.post(
                upload_url,
                headers={
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "model/gltf-binary",
                },
                content=file_content,
            )
            
            if response.status_code in [200, 201]:
                # Generate public URL
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
                return True, public_url
            else:
                error = response.text
                print(f"❌ Supabase upload error: {error}")
                return False, f"Upload failed: {error}"
                
    except Exception as e:
        print(f"❌ Exception uploading to Supabase: {e}")
        return False, str(e)


def check_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ PyTorch installed but CUDA not available")
            # Try nvidia-smi as fallback
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("⚠️ nvidia-smi works but PyTorch CUDA is not available")
                    print("💡 Try: pip install torch --index-url https://download.pytorch.org/whl/cu118")
            except:
                pass
            return False
    except ImportError:
        print("⚠️ PyTorch not installed")
        return False
    except Exception as e:
        print(f"⚠️ Error checking GPU: {e}")
        return False


def check_triposr_available() -> bool:
    """Check if TripoSR is installed."""
    run_script = Path(TRIPOSR_PATH) / "run.py"
    return run_script.exists()


# =====================================================
# API Endpoints
# =====================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = check_gpu_available()
    gpu_info = None
    
    if gpu_available:
        try:
            import torch
            gpu_info = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        except:
            pass
    
    return HealthResponse(
        status="ok",
        triposr_available=check_triposr_available(),
        gpu_available=gpu_available,
        gpu_info=gpu_info,
    )


@app.post("/generate-3d", response_model=GenerateResponse)
async def generate_3d_model(file: UploadFile = File(...)):
    """
    Generate a 3D model from an uploaded image.
    
    - Accepts: JPEG, PNG, WebP images
    - Returns: GLB model URL (uploaded to Supabase)
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    task_output_dir = os.path.join(OUTPUT_DIR, task_id)
    Path(task_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting 3D generation for task: {task_id}")
    
    try:
        # Save uploaded image
        file_ext = file.filename.split(".")[-1] if file.filename else "jpg"
        image_path = os.path.join(task_output_dir, f"input.{file_ext}")
        
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"📸 Image saved: {image_path} ({len(content)} bytes)")
        
        # Step 1: Run TripoSR
        print("🧠 Running TripoSR inference...")
        success, result = await run_triposr(image_path, task_output_dir)
        
        if not success:
            return GenerateResponse(
                success=False,
                error=result,
                task_id=task_id,
            )
        
        obj_path = result
        print(f"✅ OBJ generated: {obj_path}")
        
        # Step 2: Convert to GLB
        print("🔄 Converting to GLB...")
        success, result = await convert_obj_to_glb(obj_path)
        
        if not success:
            return GenerateResponse(
                success=False,
                error=result,
                task_id=task_id,
            )
        
        glb_path = result
        print(f"✅ GLB created: {glb_path}")
        
        # Step 3: Upload to Supabase
        print("☁️ Uploading to Supabase...")
        glb_filename = f"{task_id}.glb"
        success, result = await upload_to_supabase(glb_path, glb_filename)
        
        if not success:
            return GenerateResponse(
                success=False,
                error=result,
                task_id=task_id,
            )
        
        glb_url = result
        print(f"✅ Upload complete: {glb_url}")
        
        return GenerateResponse(
            success=True,
            glb_url=glb_url,
            task_id=task_id,
        )
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return GenerateResponse(
            success=False,
            error=str(e),
            task_id=task_id,
        )


@app.get("/generate-3d-from-url")
async def generate_3d_from_url_get(image_url: Optional[str] = None):
    """Debug endpoint - GET requests with query parameter (for proxy testing)."""
    if not image_url:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Missing image_url",
                "message": "Use POST with JSON body or GET with ?image_url=... query parameter",
            }
        )
    
    # Process as if it was a POST request
    print(f"📥 [FASTAPI] GET request with image_url: {image_url}")
    request_obj = ImageUrlRequest(image_url=image_url)
    return await generate_3d_from_url(request_obj)

@app.post("/generate-3d-from-url", response_model=GenerateResponse)
async def generate_3d_from_url(request: ImageUrlRequest):
    """
    Generate a 3D model from an image URL.
    Downloads the image and processes it.
    """
    print(f"📥 [FASTAPI] Received POST request to /generate-3d-from-url")
    print(f"📥 [FASTAPI] Request body: {request}")
    image_url = request.image_url
    task_id = str(uuid.uuid4())
    task_output_dir = os.path.join(OUTPUT_DIR, task_id)
    Path(task_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting 3D generation from URL for task: {task_id}")
    print(f"🔗 Image URL: {image_url}")
    
    try:
        # Download image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, follow_redirects=True)
            
            if response.status_code != 200:
                return GenerateResponse(
                    success=False,
                    error=f"Failed to download image: {response.status_code}",
                    task_id=task_id,
                )
            
            # Determine file extension
            content_type = response.headers.get("content-type", "image/jpeg")
            ext_map = {
                "image/jpeg": "jpg",
                "image/jpg": "jpg",
                "image/png": "png",
                "image/webp": "webp",
            }
            file_ext = ext_map.get(content_type, "jpg")
            
            image_path = os.path.join(task_output_dir, f"input.{file_ext}")
            
            with open(image_path, "wb") as f:
                f.write(response.content)
            
            print(f"📸 Image downloaded: {image_path} ({len(response.content)} bytes)")
        
        # Step 1: Run TripoSR
        print("🧠 Running TripoSR inference...")
        success, result = await run_triposr(image_path, task_output_dir)
        
        if not success:
            return GenerateResponse(
                success=False,
                error=result,
                task_id=task_id,
            )
        
        obj_path = result
        print(f"✅ OBJ generated: {obj_path}")
        
        # Step 2: Convert to GLB
        print("🔄 Converting to GLB...")
        success, result = await convert_obj_to_glb(obj_path)
        
        if not success:
            return GenerateResponse(
                success=False,
                error=result,
                task_id=task_id,
            )
        
        glb_path = result
        print(f"✅ GLB created: {glb_path}")
        
        # Step 3: Upload to Supabase
        print("☁️ Uploading to Supabase...")
        glb_filename = f"{task_id}.glb"
        success, result = await upload_to_supabase(glb_path, glb_filename)
        
        if not success:
            return GenerateResponse(
                success=False,
                error=result,
                task_id=task_id,
            )
        
        glb_url = result
        print(f"✅ Upload complete: {glb_url}")
        
        return GenerateResponse(
            success=True,
            glb_url=glb_url,
            task_id=task_id,
        )
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return GenerateResponse(
            success=False,
            error=str(e),
            task_id=task_id,
        )


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting TripoSR API Server...")
    print(f"📁 TripoSR path: {TRIPOSR_PATH}")
    print(f"📁 Output dir: {OUTPUT_DIR}")
    print(f"🔗 Supabase URL: {SUPABASE_URL or 'Not configured'}")
    print(f"🪣 Supabase bucket: {SUPABASE_BUCKET}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


