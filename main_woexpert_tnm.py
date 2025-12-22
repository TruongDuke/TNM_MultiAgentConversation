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
    get_doc_system_message_woexpert,
)
from utils.utils import prase_json, simple_retry
from utils.data import MedDataset
from utils.voting import extract_all_tnm_proposals, weighted_vote_with_confidence


def parse_args():
    parser = argparse.ArgumentParser(description="MAC-WOEXPERT TNM Setting")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_list.json",
        help="the llm models' config file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="x_llama3",
        choices=["x_llama3"],
        help="the llm models used for doctors",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tnm_cases",
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
        "--num_doctors", type=int, default=3, help="number of doctors"
    )
    parser.add_argument("--n_round", type=int, default=10, help="groupchat rounds")

    args = parser.parse_args()
    return args


@simple_retry(max_attempts=100, delay=1)
def process_single_case(
    args, dataset, idx, output_dir, model_config,
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
    identify = f"{args.num_doctors}-{args.n_round}"

    # output/MAC_WOEXPERT_TNM/tnm/x_llama3/3-10/1/...
    output_dir = osp.join(
        output_dir,
        "MAC_WOEXPERT_TNM",
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

    user_proxy = UserProxyAgent(
        name="Admin",
        system_message="A human admin doctor.",
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    Docs = []
    for index in range(args.num_doctors):
        name = f"Doctor{index}"
        # Doctor cuối cùng sẽ có instruction đặc biệt để output JSON
        is_last_doctor = (index == args.num_doctors - 1)
        doc_system_message = get_doc_system_message_woexpert(
            doctor_name=name, stage=args.stage, is_last_doctor=is_last_doctor
        )
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config,
            system_message=doc_system_message,
        )
        Docs.append(Doc)

    groupchat = GroupChat(
        agents=[user_proxy] + Docs,
        messages=[],
        max_round=args.n_round,
        speaker_selection_method="auto",
        select_speaker_auto_verbose=False,
        allow_repeat_speaker=True,
        send_introductions=False,
        max_retries_for_selecting_speaker=args.n_round // (args.num_doctors + 1),
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
    output = user_proxy.initiate_chat(
        manager,
        message=inital_message,
    )

    # Tính cost (nếu SDK hỗ trợ)
    for agent in Docs:
        try:
            case_cost += agent.client.total_usage_summary.get("total_cost", 0.0)
        except Exception:
            pass

    # Lưu toàn bộ hội thoại
    conversation_path = osp.join(output_dir, conversation_name)
    with open(conversation_path, "w") as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)

    # Extract TẤT CẢ proposals từ mọi agent
    proposals = extract_all_tnm_proposals(output.chat_history)

    if not proposals:
        raise ValueError(
            f"Không tìm thấy proposal TNM nào trong chat_history cho case {case_crl}."
        )

    # Weighted voting với confidence
    agent_weights = {}
    # Tất cả doctors có weight bằng nhau (không có Supervisor)
    for prop in proposals:
        agent_weights[prop["agent"]] = 1.0

    voting_result = weighted_vote_with_confidence(
        proposals, 
        case_presentation=case_presentation,
        agent_weights=agent_weights
    )

    # Ghi thông tin case với voting results
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

    filter_criteria = {
        "tags": [args.model_name],
    }

    config_list = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria
    )

    model_config = {
        "cache_seed": None,
        "temperature": 1,  # Temperature = 1 để có sự đa dạng trong thảo luận
        "config_list": config_list,
        "timeout": 300,
    }

    dataset = MedDataset(dataname=args.dataset_name)
    data_len = len(dataset)

    output_dir = args.output_dir

    # Chạy thử tối đa 10 case đầu để test, muốn full thì đổi min(10, data_len) -> data_len
    for idx in tqdm(range(min(10, data_len))):
        try:
            process_single_case(
                args, dataset, idx, output_dir, model_config,
            )
        except Exception as e:
            print(f"Failed to process case {idx} after all attempts: {str(e)}")
            continue


if __name__ == "__main__":
    main()

