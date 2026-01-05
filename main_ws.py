import os
import re
import time
import json
import argparse
from functools import wraps

import os.path as osp
from tqdm import tqdm

from autogen import (
    GroupChat,
    UserProxyAgent,
    ConversableAgent,
    AssistantAgent,
    GroupChatManager,
    config_list_from_json,
)

from utils.prompts import (
    get_inital_message,
    get_doc_system_message,
    get_supervisor_system_message,
    get_consultant_message,
)
from utils.utils import prase_json, simple_retry
from utils.data import MedDataset
from utils.voting import extract_all_tnm_proposals, weighted_vote_with_confidence


def parse_args():
    parser = argparse.ArgumentParser(description="MAC-WS TNM Setting")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_list.json",
        help="the llm models' config file",
    )
    parser.add_argument(
        "--query_model_name",
        type=str,
        default="x_llama3",
        choices=["x_llama3"],
        help="the llm models used for consultant (specialist selection)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="x_llama3",
        choices=["x_llama3"],
        help="the llm models used for doctors",
    )
    parser.add_argument(
        "--supervisor_model_name",
        type=str,
        default="supervisor_finetuned",
        choices=["x_llama3", "supervisor_finetuned"],
        help="the llm models used for supervisor",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tnm_cases",       # tên file dataset: dataset/tnm_cases.json
        choices=["tnm_cases"],
        help="choice dataset (TNM cases)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="tnm",
        choices=["tnm"],
        help="TNM staging (only uses initial presentation)",
    )

    parser.add_argument(
        "--times",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="repeat times for the whole experiment setting",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="log file directory",
    )
    parser.add_argument(
        "--num_specialists", type=int, default=3, help="number of experts"
    )
    parser.add_argument("--n_round", type=int, default=9, help="groupchat rounds")
    parser.add_argument("--query_round", type=int, default=1, help="consultant rounds")

    args = parser.parse_args()
    return args


@simple_retry(max_attempts=100, delay=1)
def process_single_case(
    args, dataset, idx, output_dir, model_config, query_model_config, supervisor_model_config
):
    case_cost = 0.0
    case_info = {}

    (
        case_type,
        case_name,
        case_crl,
        case_initial_presentation,
        case_follow_up_presentation,
    ) = dataset[idx]

    json_name = f"{case_crl}.json"
    conversation_name = f"{case_crl}_conversation.json"
    identify = f"{args.num_specialists}-{args.n_round}"

    # output/qwen/MAC_WS/tnm/x_llama3/3-13/1/...
    output_dir = osp.join(
        output_dir,
        "qwen",
        "MAC_WS",
        args.stage,
        args.model_name,
        identify,
        str(args.times),
    )

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    file_names = os.listdir(output_dir)
    json_files = [file for file in file_names if file.endswith(".json")]

    # Nếu đã có kết quả và conversation thì bỏ qua case này
    if json_name in json_files and conversation_name in json_files:
        return

    # TNM chỉ dùng Initial Presentation
    case_presentation = case_initial_presentation

    # 1) Coordinator & Consultant chọn chuyên khoa
    coordinator = ConversableAgent(
        "Medical_Coordinator",
        system_message=(
            "You are a Medical Coordinator. Your role is to provide the patient's "
            "medical history and ask questions to determine the appropriate specialist. "
            "You should seek clarification and ensure all relevant information is covered."
        ),
        llm_config=query_model_config,
        human_input_mode="NEVER",
    )

    consultant = ConversableAgent(
        "Senior_Medical_Consultant",
        system_message=(
            "You are a Senior Medical Consultant. Your role is to answer the Medical "
            "Coordinator's questions, recommend the appropriate specialist based on the "
            "medical history provided, and correct any misconceptions."
        ),
        llm_config=query_model_config,
        human_input_mode="NEVER",
    )

    consultant_message = get_consultant_message(
        case_presentation, int(args.num_specialists)
    )

    result = coordinator.initiate_chat(
        consultant, message=consultant_message, max_turns=args.query_round
    )

    top_k_specialists = prase_json(result.chat_history[-1]["content"])[
        "top_k_specialists"
    ]
    assert len(top_k_specialists) == args.num_specialists
    # Nếu Autogen có trường cost thì cộng, còn không có thì bỏ qua
    try:
        case_cost += result.cost["usage_including_cached_inference"]["total_cost"]
    except Exception:
        pass

    # 2) Tạo các bác sĩ chuyên khoa
    Docs = []
    for specialist in top_k_specialists:
        name = specialist.replace(" ", "_")
        doc_system_message = get_doc_system_message(
            doctor_name=name, stage=args.stage
        )

        Doc = AssistantAgent(
            name=name,
            llm_config=model_config,
            system_message=doc_system_message,
        )
        Docs.append(Doc)

    # 3) Supervisor (TNM Staging Supervisor)
    supervisor_system_message = get_supervisor_system_message(
        stage=args.stage, use_specialist=True, specialists=top_k_specialists
    )

    Supervisor = AssistantAgent(
        name="Supervisor",
        llm_config=supervisor_model_config,
        system_message=supervisor_system_message,
    )

    agents = Docs + [Supervisor]

    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=args.n_round,
        speaker_selection_method="auto",
        admin_name="Critic",
        select_speaker_auto_verbose=False,
        allow_repeat_speaker=True,
        send_introductions=False,
        max_retries_for_selecting_speaker=args.n_round // (1 + args.num_specialists),
    )

    time.sleep(5)

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=model_config,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    inital_message = get_inital_message(
        patient_history=case_presentation, stage=args.stage
    )

    output = Supervisor.initiate_chat(
        manager,
        message=inital_message,
    )

    # 4) Tính cost (nếu SDK hỗ trợ)
    for agent in agents:
        try:
            case_cost += agent.client.total_usage_summary.get("total_cost", 0.0)
        except Exception:
            pass

    # 5) Lưu toàn bộ hội thoại
    conversation_path = osp.join(output_dir, conversation_name)
    with open(conversation_path, "w") as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)

    # 6) Extract TẤT CẢ proposals từ mọi agent
    proposals = extract_all_tnm_proposals(output.chat_history)

    if not proposals:
        raise ValueError(
            f"Không tìm thấy proposal TNM nào trong chat_history cho case {case_crl}."
        )

    # 7) Weighted voting với confidence
    agent_weights = {}
    # Supervisor có weight cao hơn
    for prop in proposals:
        if "Supervisor" in prop["agent"]:
            agent_weights[prop["agent"]] = 1.5
        else:
            agent_weights[prop["agent"]] = 1.0

    voting_result = weighted_vote_with_confidence(
        proposals, 
        case_presentation=case_presentation,
        agent_weights=agent_weights
    )

    # 8) Ghi thông tin case với voting results
    case_info["Type"] = case_type
    case_info["Crl"] = case_crl
    case_info["Cost"] = case_cost
    case_info["Presentation"] = case_presentation
    case_info["Name"] = case_name

    # Kết quả TNM từ voting
    case_info["T"] = voting_result["T"]
    case_info["N"] = voting_result["N"]
    case_info["M"] = voting_result["M"]
    case_info["TNM"] = voting_result["TNM"]

    # Thông tin voting (MỚI)
    case_info["Consensus_Score"] = voting_result["consensus_score"]
    case_info["Factor_Consensus"] = voting_result["factor_consensus"]
    case_info["Confidence"] = voting_result["confidence"]
    case_info["Disagreements"] = voting_result["disagreements"]
    case_info["Num_Proposals"] = voting_result["num_proposals"]

    # Rationale từ proposal có confidence cao nhất
    best_proposal = max(
        proposals,
        key=lambda p: sum(1 for c in p.get("combined_confidence", {}).values() if c == "high")
    )
    case_info["Rationale"] = best_proposal.get("rationale", "")
    case_info["Areas of Disagreement"] = ", ".join(voting_result["disagreements"])

    recorder_path = osp.join(output_dir, json_name)
    with open(recorder_path, "w") as file:
        json.dump(case_info, file, indent=4, ensure_ascii=False)


def main():
    args = parse_args()

    query_filter_criteria = {
        "tags": [args.query_model_name],
    }

    filter_criteria = {
        "tags": [args.model_name],
    }

    supervisor_filter_criteria = {
        "tags": [args.supervisor_model_name],
    }

    query_config_list = config_list_from_json(
        env_or_file=args.config, filter_dict=query_filter_criteria
    )

    config_list = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria
    )

    supervisor_config_list = config_list_from_json(
        env_or_file=args.config, filter_dict=supervisor_filter_criteria
    )

    query_model_config = {
        "cache_seed": None,
        "temperature": 0,
        "config_list": query_config_list,
        "timeout": 120,
    }

    model_config = {
        "cache_seed": None,
        "temperature": 1,
        "config_list": config_list,
        "timeout": 300,
    }

    supervisor_model_config = {
        "cache_seed": None,
        "temperature": 1,
        "config_list": supervisor_config_list,
        "timeout": 300,
    }

    dataset = MedDataset(dataname=args.dataset_name)
    data_len = len(dataset)

    output_dir = args.output_dir

    # Chạy toàn bộ dataset
    for idx in tqdm(range(data_len)):
        try:
            process_single_case(
                args, dataset, idx, output_dir, model_config, query_model_config, supervisor_model_config
            )
        except Exception as e:
            print(f"Failed to process case {idx} after all attempts: {str(e)}")
            continue


if __name__ == "__main__":
    main()