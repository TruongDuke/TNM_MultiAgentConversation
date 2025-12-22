import json
import os.path as osp


class MedDataset:
    """
    Dataset loader cho các case y khoa.

    - Mặc định đọc file: dataset/tnm_cases.json
    - Cấu trúc file JSON:
        {
          "Cases": [
            {
              "Type": "TNM",
              "Final Name": "{T:T1,N:N1,M:M0}",
              "Case URL": "133166",
              "Initial Presentation": "...",
              "Follow-up Presentation": "",
              "Meta": {
                "split": "Train",
                "T": 1,
                "N": 1,
                "M": 0
              }
            },
            ...
          ]
        }
    """

    dataset_dir = "dataset"  # thư mục chứa data

    def __init__(self, dataname: str = "tnm_cases"):
        """
        dataname: tên file (không kèm .json), ví dụ:
          - "tnm_cases"               -> dataset/tnm_cases.json
          - "rare_disease_cases_302"  -> dataset/rare_disease_cases_302.json
        """
        filename = f"{dataname}.json"
        self.data_path = osp.join(self.dataset_dir, filename)
        self.cases = None
        self.load()

    def load(self):
        """Đọc dữ liệu từ file JSON vào self.cases"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # file phải có key "Cases"
        self.cases = data["Cases"]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        """
        Trả về 5 giá trị giống code gốc:
          (disease_type, disease_name, disease_crl,
           disease_initial_presentation, disease_follow_up_presentation)
        """
        case = self.cases[idx]

        disease_type = case["Type"]
        disease_name = case["Final Name"]
        disease_crl = case["Case URL"]
        disease_initial_presentation = case["Initial Presentation"]
        disease_follow_up_presentation = case["Follow-up Presentation"]

        return (
            disease_type,
            disease_name,
            disease_crl,
            disease_initial_presentation,
            disease_follow_up_presentation,
        )

    def get_meta(self, idx: int):
        """
        Lấy thông tin Meta (nếu có), ví dụ với TNM:
          {"split": "Train", "T": 1, "N": 1, "M": 0}
        """
        case = self.cases[idx]
        return case.get("Meta", {})