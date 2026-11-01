import os
import sys
import time
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://zenodo.org/api/records/7260705/files/Task1.zip/content"
TOTAL_SIZE = 14471900926 # ~13.48 GB
DEST_DIR = "/Volumes/Jonath SSD1/Research/Quantum Fusion"
ZIP_PATH = os.path.join(DEST_DIR, "Task1.zip")
PROGRESS_PATH = os.path.join(DEST_DIR, "Task1_progress.json")
CHUNK_SIZE = 16 * 1024 * 1024 # 16 MB chunks
MAX_WORKERS = 8 # 8 parallel HTTP streams

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, "r") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def save_progress(completed_chunks):
    try:
        tmp_path = PROGRESS_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(list(completed_chunks), f)
        os.replace(tmp_path, PROGRESS_PATH)
    except Exception as e:
        print(f"\n[Warning] Progress save failed: {e}")

def download_chunk(chunk_idx, start, end, max_retries=6):
    expected_len = end - start + 1
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = bytearray()
                while len(data) < expected_len:
                    part = resp.read(min(1024 * 1024, expected_len - len(data)))
                    if not part:
                        break
                    data.extend(part)
                    
                if len(data) != expected_len:
                    raise IOError(f"Truncated chunk: expected {expected_len}, got {len(data)}")
                
                with open(ZIP_PATH, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return chunk_idx, expected_len
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[Error] Chunk {chunk_idx} failed after {max_retries} attempts: {e}")
                return chunk_idx, 0
            time.sleep(2.0 * (attempt + 1))

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print("=" * 80)
    print("--- SYNTHRAD2023 TASK 1 RESUMABLE MULTI-STREAM DOWNLOADER ---")
    print(f"Target: {ZIP_PATH}")
    print(f"Total Size: {TOTAL_SIZE / (1024**3):.2f} GB ({TOTAL_SIZE:,} bytes)")
    print(f"Chunk Size: {CHUNK_SIZE / (1024**2):.1f} MB | Concurrency: {MAX_WORKERS} streams")
    print("=" * 80)
    
    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) != TOTAL_SIZE:
        print("[Allocation] Initializing target file container on SSD...")
        with open(ZIP_PATH, "wb") as f:
            f.truncate(TOTAL_SIZE)
        print("[Allocation] Container initialized.")
        
    chunks = []
    chunk_idx = 0
    for start in range(0, TOTAL_SIZE, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, TOTAL_SIZE - 1)
        chunks.append((chunk_idx, start, end))
        chunk_idx += 1
        
    total_chunks = len(chunks)
    completed_chunks = load_progress()
    print(f"[State] Total chunks: {total_chunks} | Already downloaded: {len(completed_chunks)} ({len(completed_chunks)*CHUNK_SIZE/(1024**3):.2f} GB)")
    
    pending_chunks = [c for c in chunks if c[0] not in completed_chunks]
    
    if pending_chunks:
        print(f"[Start] Downloading {len(pending_chunks)} chunks across {MAX_WORKERS} threads...\n")
        session_bytes = 0
        t0 = time.time()
        last_print = t0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_chunk, c[0], c[1], c[2]): c[0] for c in pending_chunks}
            
            for future in as_completed(futures):
                c_idx, bytes_read = future.result()
                if bytes_read > 0:
                    completed_chunks.add(c_idx)
                    save_progress(completed_chunks)
                    session_bytes += bytes_read
                    
                    now = time.time()
                    if now - last_print >= 2.0 or len(completed_chunks) == total_chunks:
                        elapsed = now - t0
                        speed_mb = (session_bytes / (1024**2)) / max(0.1, elapsed)
                        downloaded_bytes = len(completed_chunks) * CHUNK_SIZE
                        progress_pct = min(100.0, (downloaded_bytes / TOTAL_SIZE) * 100)
                        remaining_bytes = max(0, TOTAL_SIZE - downloaded_bytes)
                        eta_min = (remaining_bytes / max(1, (session_bytes / max(0.1, elapsed)))) / 60
                        
                        bar_len = 30
                        filled = int(bar_len * progress_pct / 100)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        
                        print(
                            f"[{bar}] {progress_pct:5.1f}% | "
                            f"{downloaded_bytes/(1024**3):.2f}/{TOTAL_SIZE/(1024**3):.2f} GB | "
                            f"{speed_mb:5.2f} MB/s | ETA: {eta_min:4.1f} min",
                            flush=True
                        )
                        last_print = now
                        
    print("\n" + "=" * 80)
    if len(completed_chunks) == total_chunks:
        print("[SUCCESS] Download verified 100% complete! Proceeding to extract...")
        extract_dir = os.path.join(DEST_DIR, "data")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[Extraction] Unzipping Task1.zip into: {extract_dir}...")
        t_unzip = time.time()
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"[Extraction] Finished in {time.time() - t_unzip:.1f}s.")
        print("[Status] All 181 patient scans ready at:")
        print(f"         {os.path.join(extract_dir, 'Task1', 'brain')}")
    else:
        print(f"[PARTIAL] Download interrupted: {len(completed_chunks)}/{total_chunks} chunks.")
    print("=" * 80)

if __name__ == "__main__":
    main()
