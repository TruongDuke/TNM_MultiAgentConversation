"""
Confidence scoring for TNM staging.
Calculates confidence based on case description clarity and agent agreement.
"""
import re
from collections import Counter


def calculate_confidence_from_case(proposal, case_presentation):
    """
    Tính confidence dựa trên độ rõ ràng của thông tin trong case description.
    
    Args:
        proposal: Dict với keys T, N, M
        case_presentation: Case description text
        
    Returns:
        Dict với keys:
        - confidence: dict với T, N, M confidence levels ("high", "medium", "low")
        - reasons: dict với lý do cho mỗi factor
        - overall: overall confidence level
    """
    confidence = {"T": "medium", "N": "medium", "M": "medium"}
    reasons = {"T": [], "N": [], "M": []}
    
    # T factor: Check size và invasion
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|mm|㎝)', case_presentation, re.IGNORECASE)
    if size_match:
        confidence["T"] = "high"
        reasons["T"].append("Tumor size explicitly mentioned")
    else:
        confidence["T"] = "low"
        reasons["T"].append("Tumor size not mentioned")
    
    # Check invasion
    invasion_keywords = ['mediastinum', 'trachea', 'heart', 'chest wall', 'invasion', 
                        '浸潤', '縦隔', '気管', '心臓', '胸壁']
    if any(kw.lower() in case_presentation.lower() for kw in invasion_keywords):
        if confidence["T"] == "low":
            confidence["T"] = "medium"
        reasons["T"].append("Invasion mentioned")
    
    # N factor: Check lymph nodes
    lymph_keywords = ['lymph node', 'リンパ節', 'hilar', 'mediastinal', 'scalene', 
                     'supraclavicular', '肺門', '縦隔', '鎖骨上']
    has_lymph = any(kw.lower() in case_presentation.lower() for kw in lymph_keywords)
    
    if has_lymph:
        # Check location clarity
        location_keywords = ['ipsilateral', 'contralateral', '同側', '対側', 
                           '同側縦隔', '対側縦隔']
        if any(kw.lower() in case_presentation.lower() for kw in location_keywords):
            confidence["N"] = "high"
            reasons["N"].append("Lymph node location clearly specified")
        else:
            confidence["N"] = "medium"
            reasons["N"].append("Lymph nodes mentioned but location ambiguous")
    else:
        confidence["N"] = "low"
        reasons["N"].append("No clear lymph node information")
    
    # M factor: Check metastasis
    metastasis_keywords = ['metastasis', '転移', 'distant', 'bone', 'liver', 'brain', 
                          'adrenal', '骨転移', '遠隔転移', '肝転移', '脳転移', '副腎']
    has_metastasis = any(kw.lower() in case_presentation.lower() for kw in metastasis_keywords)
    
    if has_metastasis:
        confidence["M"] = "high"
        reasons["M"].append("Distant metastasis explicitly mentioned")
    elif 'no distant' in case_presentation.lower() or '胸水認めません' in case_presentation:
        confidence["M"] = "high"
        reasons["M"].append("Explicitly stated no distant metastasis")
    else:
        # Check for negative statements
        negative_keywords = ['認めません', 'ありません', 'ない', 'no']
        if any(kw in case_presentation for kw in negative_keywords):
            confidence["M"] = "medium"
            reasons["M"].append("Negative statement about metastasis")
        else:
            confidence["M"] = "medium"
            reasons["M"].append("No explicit mention of distant metastasis")
    
    return {
        "confidence": confidence,
        "reasons": reasons,
        "overall": "high" if all(c == "high" for c in confidence.values()) 
                  else "low" if any(c == "low" for c in confidence.values())
                  else "medium"
    }


def calculate_confidence_from_agreement(proposal, all_proposals):
    """
    Tính confidence dựa trên sự đồng thuận với các agents khác.
    
    Args:
        proposal: Dict với keys agent, T, N, M
        all_proposals: List of all proposals
        
    Returns:
        Dict với keys:
        - T, N, M: confidence levels
        - agreement_rates: dict với agreement rates cho mỗi factor
    """
    if not all_proposals or len(all_proposals) <= 1:
        return {"T": "medium", "N": "medium", "M": "medium"}
    
    # Loại bỏ proposal hiện tại
    other_proposals = [p for p in all_proposals 
                      if p["agent"] != proposal["agent"]]
    
    if not other_proposals:
        return {"T": "medium", "N": "medium", "M": "medium"}
    
    # Tính agreement rate
    t_values = [p["T"] for p in other_proposals]
    n_values = [p["N"] for p in other_proposals]
    m_values = [p["M"] for p in other_proposals]
    
    t_agreement = t_values.count(proposal["T"]) / len(t_values) if t_values else 0
    n_agreement = n_values.count(proposal["N"]) / len(n_values) if n_values else 0
    m_agreement = m_values.count(proposal["M"]) / len(m_values) if m_values else 0
    
    # Map agreement rate to confidence
    def agreement_to_conf(rate):
        if rate >= 0.8:
            return "high"
        elif rate >= 0.5:
            return "medium"
        else:
            return "low"
    
    return {
        "T": agreement_to_conf(t_agreement),
        "N": agreement_to_conf(n_agreement),
        "M": agreement_to_conf(m_agreement),
        "agreement_rates": {
            "T": t_agreement,
            "N": n_agreement,
            "M": m_agreement
        }
    }

