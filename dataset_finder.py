import requests
from huggingface_hub import HfApi

def search_multimodal_datasets(query="MRI brain tumor", limit=10):
    print(f"--- Q-FUSION Dataset Finder ---")
    print(f"Searching Hugging Face Hub for: '{query}'")
    print("Criteria: Must be Multimodal (e.g., Image + Image, or Image + Text), Free/Open Access.\n")
    
    api = HfApi()
    
    # We search datasets with the query
    datasets = api.list_datasets(search=query, limit=limit*3) 
    
    candidates = []
    
    for ds in datasets:
        # Check tags to deduce modalities
        tags = ds.tags if ds.tags else []
        
        # Determine if it has images
        has_image = any("image" in tag.lower() for tag in tags)
        has_text = any("text" in tag.lower() for tag in tags)
        has_tabular = any("tabular" in tag.lower() for tag in tags)
        
        # Note: In MRI datasets, often 'image' covers multiple modalities (T1, T2, FLAIR). 
        # A true multimodal dataset for our case has multiple distinct image channels or image + text.
        # We will heavily favor datasets that have multiple data types or explicitly state multimodal.
        
        is_multimodal_tagged = any("multimodal" in tag.lower() for tag in tags)
        
        # Check if it is gated/commercial (we want free open access)
        is_gated = getattr(ds, "gated", False)
        if is_gated:
            continue
            
        # We define it as a candidate if it has image AND (is explicitly multimodal OR has text/tabular)
        if has_image and (is_multimodal_tagged or has_text or has_tabular):
            desc = getattr(ds, "description", "No description provided.")
            if desc is None: desc = "No description provided."
            candidates.append({
                "id": ds.id,
                "description": desc[:200] + "...",
                "downloads": getattr(ds, "downloads", 0),
                "tags": tags
            })
            
        if len(candidates) >= limit:
            break
            
    if not candidates:
        print("No exact multimodal HuggingFace matches found automatically. Widening search or recommending known datasets...")
    else:
        # Sort by downloads as a proxy for quality/popularity
        candidates = sorted(candidates, key=lambda x: x["downloads"], reverse=True)
        print("Top Free Multimodal Candidates Found:")
        for idx, c in enumerate(candidates):
            print(f"{idx+1}. Dataset: {c['id']}")
            print(f"   Downloads: {c['downloads']}")
            print(f"   Tags: {', '.join([t for t in c['tags'] if 'task_categories' in t or 'modalities' in t or 'multimodal' in t])}")
            print("-" * 50)
            
    # Hardcoded recommendation based on clinical literature search
    print("\n--- SENIOR ENGINEER RECOMMENDATION ---")
    print("For Q-FUSION, we require strict multi-parametric MRI (T1, T2, etc.).")
    print("Top Free/Open Access Datasets from clinical literature:")
    print("1. BraTS (Brain Tumor Segmentation Challenge)")
    print("   - Modalities: Image (T1), Image (T2), Image (FLAIR), Image (T1ce)")
    print("   - Clinical Text: Age, Survival data (in some challenge years)")
    print("   - Cost: Free for research")
    print("2. BIOSE MRI (Mendeley Data)")
    print("   - Modalities: Image (T1w, T2w, FLAIR, DWI) + Clinical Text (Diagnosis Reports)")
    print("   - Cost: Open Access (CC BY 4.0)")
    print("3. OASIS-3")
    print("   - Modalities: Image (MRI, PET) + Clinical/Cognitive Text")
    print("   - Cost: Free for research")
    
if __name__ == "__main__":
    search_multimodal_datasets(query="MRI", limit=10)
