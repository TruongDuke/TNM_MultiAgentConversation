"""
Voting mechanism for TNM staging.
Extracts all proposals from agents and performs weighted voting with confidence.
"""
import re
from collections import Counter, defaultdict
from utils.utils import prase_json


def extract_all_tnm_proposals(chat_history):
    """
    Trích xuất TẤT CẢ các đề xuất T, N, M từ MỌI agent trong chat history.
    
    Args:
        chat_history: List of messages from AutoGen chat
        
    Returns:
        List of proposals, mỗi proposal có keys:
        - agent: tên agent
        - T, N, M: giá trị TNM
        - TNM: string format
        - rationale: lý do
        - confidence: confidence dict (nếu có trong JSON)
        - message_index: vị trí trong chat
        - source: "json" hoặc "regex"
    """
    proposals = []
    
    for idx, msg in enumerate(chat_history):
        agent_name = msg.get("name", "unknown")
        content = msg.get("content", "")
        
        # Thử parse JSON trước (ưu tiên)
        try:
            json_data = prase_json(content)
            if "T" in json_data and "N" in json_data and "M" in json_data:
                proposals.append({
                    "agent": agent_name,
                    "T": json_data.get("T"),
                    "N": json_data.get("N"),
                    "M": json_data.get("M"),
                    "TNM": json_data.get("TNM", 
                        f"T{json_data.get('T')},N{json_data.get('N')},M{json_data.get('M')}"),
                    "rationale": json_data.get("Rationale", ""),
                    "confidence": json_data.get("Confidence", {}),
                    "message_index": idx,
                    "source": "json"
                })
                continue
        except:
            pass
        
        # Fallback: tìm bằng regex
        t_match = re.search(r'\bT[:\s]*([0-4])', content, re.IGNORECASE)
        n_match = re.search(r'\bN[:\s]*([0-3])', content, re.IGNORECASE)
        m_match = re.search(r'\bM[:\s]*([01])', content, re.IGNORECASE)
        
        if t_match and n_match and m_match:
            proposals.append({
                "agent": agent_name,
                "T": int(t_match.group(1)),
                "N": int(n_match.group(1)),
                "M": int(m_match.group(1)),
                "TNM": f"T{t_match.group(1)},N{n_match.group(1)},M{m_match.group(1)}",
                "rationale": content[:300],  # Lấy 300 ký tự đầu làm rationale
                "confidence": {},
                "message_index": idx,
                "source": "regex"
            })
    
    return proposals


def weighted_vote_with_confidence(proposals, case_presentation=None, 
                                  agent_weights=None):
    """
    Weighted voting với confidence scoring.
    
    Args:
        proposals: List of proposals từ extract_all_tnm_proposals()
        case_presentation: Case description để tính confidence (optional)
        agent_weights: Dict {agent_name: weight}, default: Supervisor=1.5, others=1.0
    
    Returns:
        Dict với keys:
        - T, N, M: final values
        - TNM: string format
        - consensus_score: overall consensus (0-1)
        - factor_consensus: dict với T, N, M consensus scores
        - confidence: aggregated confidence
        - disagreements: list of disagreements
        - vote_distribution: vote counts for each factor
        - num_proposals: số lượng proposals
        - proposals: list of all proposals (với confidence đã tính)
    """
    if not proposals:
        return None
    
    # Import confidence functions
    from utils.confidence import (
        calculate_confidence_from_case,
        calculate_confidence_from_agreement
    )
    
    # Default weights
    if agent_weights is None:
        agent_weights = {}
        for prop in proposals:
            if "Supervisor" in prop["agent"]:
                agent_weights[prop["agent"]] = 1.5
            else:
                agent_weights[prop["agent"]] = 1.0
    
    # Tính confidence cho mỗi proposal
    for prop in proposals:
        # Confidence từ case description
        if case_presentation:
            try:
                case_conf = calculate_confidence_from_case(prop, case_presentation)
                prop["case_confidence"] = case_conf["confidence"]
                prop["case_confidence_reasons"] = case_conf["reasons"]
            except Exception as e:
                # Nếu lỗi, dùng default
                prop["case_confidence"] = {"T": "medium", "N": "medium", "M": "medium"}
                prop["case_confidence_reasons"] = {}
        
        # Confidence từ agreement
        try:
            agreement_conf = calculate_confidence_from_agreement(prop, proposals)
            prop["agreement_confidence"] = agreement_conf
        except Exception as e:
            prop["agreement_confidence"] = {"T": "medium", "N": "medium", "M": "medium"}
        
        # Combined confidence (ưu tiên case confidence)
        combined_conf = {}
        for factor in ["T", "N", "M"]:
            case_c = prop.get("case_confidence", {}).get(factor, "medium")
            agree_c = prop.get("agreement_confidence", {}).get(factor, "medium")
            
            # Nếu cả 2 đều high → high, nếu 1 low → low, còn lại medium
            if case_c == "high" and agree_c == "high":
                combined_conf[factor] = "high"
            elif case_c == "low" or agree_c == "low":
                combined_conf[factor] = "low"
            else:
                combined_conf[factor] = "medium"
        
        prop["combined_confidence"] = combined_conf
    
    # Weighted voting với confidence multiplier
    t_votes = defaultdict(float)
    n_votes = defaultdict(float)
    m_votes = defaultdict(float)
    
    for prop in proposals:
        agent = prop["agent"]
        base_weight = agent_weights.get(agent, 1.0)
        
        # Confidence multiplier
        conf_multiplier = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.5
        }
        
        t_weight = base_weight * conf_multiplier.get(
            prop["combined_confidence"].get("T", "medium"), 1.0
        )
        n_weight = base_weight * conf_multiplier.get(
            prop["combined_confidence"].get("N", "medium"), 1.0
        )
        m_weight = base_weight * conf_multiplier.get(
            prop["combined_confidence"].get("M", "medium"), 1.0
        )
        
        t_votes[prop["T"]] += t_weight
        n_votes[prop["N"]] += n_weight
        m_votes[prop["M"]] += m_weight
    
    # Chọn giá trị có vote cao nhất
    final_t = max(t_votes.items(), key=lambda x: x[1])[0]
    final_n = max(n_votes.items(), key=lambda x: x[1])[0]
    final_m = max(m_votes.items(), key=lambda x: x[1])[0]
    
    # Tính consensus scores
    total_weight = sum(agent_weights.values())
    max_possible_weight = total_weight * 1.5  # Max với high confidence
    t_consensus = t_votes[final_t] / max_possible_weight if max_possible_weight > 0 else 0
    n_consensus = n_votes[final_n] / max_possible_weight if max_possible_weight > 0 else 0
    m_consensus = m_votes[final_m] / max_possible_weight if max_possible_weight > 0 else 0
    overall_consensus = (t_consensus + n_consensus + m_consensus) / 3
    
    # Tìm disagreements
    disagreements = []
    if t_consensus < 0.7:
        disagree_count = len([p for p in proposals if p["T"] != final_t])
        disagreements.append(f"T: {disagree_count}/{len(proposals)} agents disagree")
    if n_consensus < 0.7:
        disagree_count = len([p for p in proposals if p["N"] != final_n])
        disagreements.append(f"N: {disagree_count}/{len(proposals)} agents disagree")
    if m_consensus < 0.7:
        disagree_count = len([p for p in proposals if p["M"] != final_m])
        disagreements.append(f"M: {disagree_count}/{len(proposals)} agents disagree")
    
    # Aggregate confidence
    all_confidences = [p.get("combined_confidence", {}) for p in proposals]
    conf_counter = {
        "T": Counter([c.get("T", "medium") for c in all_confidences]),
        "N": Counter([c.get("N", "medium") for c in all_confidences]),
        "M": Counter([c.get("M", "medium") for c in all_confidences])
    }
    
    def get_aggregate_conf(counter):
        if "low" in counter and counter["low"] >= len(proposals) * 0.5:
            return "low"
        elif "high" in counter and counter["high"] >= len(proposals) * 0.7:
            return "high"
        else:
            return "medium"
    
    aggregated_confidence = {
        "T": get_aggregate_conf(conf_counter["T"]),
        "N": get_aggregate_conf(conf_counter["N"]),
        "M": get_aggregate_conf(conf_counter["M"])
    }
    
    return {
        "T": final_t,
        "N": final_n,
        "M": final_m,
        "TNM": f"T{final_t},N{final_n},M{final_m}",
        "consensus_score": overall_consensus,
        "factor_consensus": {
            "T": t_consensus,
            "N": n_consensus,
            "M": m_consensus
        },
        "confidence": aggregated_confidence,
        "disagreements": disagreements if disagreements else ["None"],
        "vote_distribution": {
            "T": dict(t_votes),
            "N": dict(n_votes),
            "M": dict(m_votes)
        },
        "num_proposals": len(proposals),
        "proposals": proposals  # Giữ lại để debug/analysis
    }

