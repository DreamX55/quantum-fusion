import os
import sys
import json
import time
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HF_REPO_API = "https://huggingface.co/api/datasets/rocky93/BraTS_segmentation"
HF_RAW_BASE = "https://huggingface.co/datasets/rocky93/BraTS_segmentation/resolve/main"
MODALITIES = ["flair", "t1", "t1ce", "t2", "seg"]

def get_subject_list():
    req = urllib.request.Request(HF_REPO_API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        
    siblings = [s["rfilename"] for s in data.get("siblings", [])]
    subjects = sorted(list(set([s.split("/")[0] for s in siblings if "/" in s])))
    return subjects

def download_file(url, dest_path, retries=3):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True, 0
        
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                chunk_size = 64 * 1024
                bytes_downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    bytes_downloaded += len(chunk)
            return True, bytes_downloaded
        except Exception as e:
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            if attempt == retries - 1:
                return False, str(e)
            time.sleep(1.0)
    return False, "Max retries exceeded"

def download_subject(sub_id, output_dir):
    sub_dir = os.path.join(output_dir, sub_id)
    os.makedirs(sub_dir, exist_ok=True)
    
    total_bytes = 0
    for mod in MODALITIES:
        fname = f"{sub_id}_{mod}.nii.gz"
        url = f"{HF_RAW_BASE}/{sub_id}/{fname}"
        dest = os.path.join(sub_dir, fname)
        
        success, res = download_file(url, dest)
        if not success:
            return False, sub_id, f"Failed {fname}: {res}"
        total_bytes += res if isinstance(res, int) else 0
        
    return True, sub_id, total_bytes

def main():
    parser = argparse.ArgumentParser(description="Download 4-Modality BraTS2021 Cohort (T1, T1ce, T2, FLAIR, Seg)")
    parser.add_argument("--output_dir", type=str, default="/Volumes/Jonath SSD1/Research/Quantum Fusion/data/brats2021", help="Target output directory on SSD")
    parser.add_argument("--num_patients", type=int, default=150, help="Number of subjects to download")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download worker threads")
    args = parser.parse_args()
    
    print("=" * 80)
    print("BraTS2021 4-MODALITY DATASET INGESTION PIPELINE")
    print(f"Target Directory: {args.output_dir}")
    print(f"Cohort Target:    {args.num_patients} subjects")
    print(f"Modalities:       T1, T1ce, T2, FLAIR + Ground-Truth Seg (5 files/subject)")
    print(f"Parallel Workers: {args.workers}")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nFetching catalog from Hugging Face Hub (rocky93/BraTS_segmentation)...")
    subjects = get_subject_list()
    print(f"Total available subjects in catalog: {len(subjects)}")
    
    selected_subjects = subjects[:args.num_patients]
    print(f"Selected cohort of {len(selected_subjects)} subjects: {selected_subjects[0]} to {selected_subjects[-1]}")
    
    # Check already completed
    existing = 0
    for s in selected_subjects:
        s_dir = os.path.join(args.output_dir, s)
        if os.path.exists(s_dir):
            if all(os.path.exists(os.path.join(s_dir, f"{s}_{m}.nii.gz")) for m in MODALITIES):
                existing += 1
    if existing > 0:
        print(f"Found {existing}/{len(selected_subjects)} subjects already downloaded.")
        
    t0 = time.time()
    completed = 0
    total_bytes = 0
    failed = []
    
    print(f"\nInitiating parallel download with {args.workers} threads...\n")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_subject, s, args.output_dir): s for s in selected_subjects}
        for future in as_completed(futures):
            sub_id = futures[future]
            try:
                success, sid, res = future.result()
                if success:
                    completed += 1
                    total_bytes += res
                    if completed % 10 == 0 or completed == len(selected_subjects):
                        elapsed = time.time() - t0
                        rate = (total_bytes / (1024 * 1024)) / max(elapsed, 0.001)
                        print(f"[{completed:03d}/{len(selected_subjects):03d}] Downloaded {sid} | Total: {total_bytes / (1024*1024):.1f} MB | Speed: {rate:.2f} MB/s | Elapsed: {elapsed:.1f}s")
                else:
                    failed.append((sid, res))
                    print(f"[FAIL] {sid}: {res}")
            except Exception as exc:
                failed.append((sub_id, str(exc)))
                print(f"[ERROR] {sub_id} generated exception: {exc}")
                
    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print(f"Successfully downloaded: {completed}/{len(selected_subjects)} subjects")
    print(f"Total data transferred:  {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.2f} GB)")
    print(f"Total time elapsed:      {total_time:.1f} seconds ({total_time / 60:.2f} minutes)")
    if failed:
        print(f"Failed subjects ({len(failed)}): {failed}")
    print("=" * 80)
    
    # Setup local workspace symlink
    local_data_dir = os.path.join(os.getcwd(), "data", "brats2021")
    os.makedirs(os.path.dirname(local_data_dir), exist_ok=True)
    if not os.path.exists(local_data_dir):
        try:
            os.symlink(args.output_dir, local_data_dir)
            print(f"Created symlink: {local_data_dir} -> {args.output_dir}")
        except Exception as e:
            print(f"Notice on symlink creation: {e}")
            
if __name__ == "__main__":
    main()
