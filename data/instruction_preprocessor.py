import pandas as pd
import json

class InstructionDataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.df = None

    def load_json(self):
        """JSON 파일을 로드해서 데이터 저장"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def process_data(self):
        """instruction을 추출해 DataFrame 생성"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다. 먼저 load_json()을 호출하세요.")
        self.df = pd.DataFrame(
            [{'instruction': item['instruction'], 'label': 0} for item in self.data]
        )

    def save_to_csv(self, output_path: str):
        """DataFrame을 CSV 파일로 저장"""
        if self.df is None:
            raise ValueError("DataFrame이 생성되지 않았습니다. 먼저 process_data()를 호출하세요.")
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')

    def get_dataframe(self):
        """생성된 DataFrame 반환"""
        if self.df is None:
            raise ValueError("DataFrame이 생성되지 않았습니다. 먼저 process_data()를 호출하세요.")
        return self.df
    
processor = InstructionDataProcessor('raw/alpaca_data.json')
processor.load_json()
processor.process_data()
processor.save_to_csv('preprocessed/instructions.csv')