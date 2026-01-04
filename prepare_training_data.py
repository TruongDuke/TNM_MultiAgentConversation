import json

def prepare_training_data(input_file="dataset/tnm_cases.json", output_file="dataset/tnm_training_data.json"):
    """Convert tnm_cases.json sang format training."""
    print(f"📖 Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    training_data = []
    
    for case in data["Cases"]:
        initial_presentation = case.get("Initial Presentation", "").strip()
        meta = case.get("Meta", {})
        
        t = meta.get("T")
        n = meta.get("N")
        m = meta.get("M")
        
        if t is None or n is None or m is None or not initial_presentation:
            continue
        
        tnm = f"T{t},N{n},M{m}"
        output = f"TNM: {tnm}\nT = {t}, N = {n}, M = {m}"
        
        training_data.append({
            "instruction": "Analyze this thoracic oncology case and determine the TNM cancer stage according to the TNM classification for lung cancer. Provide T, N, M values.",
            "input": initial_presentation,
            "output": output
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(training_data)} examples to {output_file}")
    return training_data

if __name__ == "__main__":
    prepare_training_data()

