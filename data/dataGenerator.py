import random
from datetime import datetime

# 전화번호 생성기
class PhoneNumberGenerator:
    def __init__(self, count):
        self.count = count  # 생성할 전화번호 개수

    def generate(self):
        phone_numbers = []
        for _ in range(self.count):
            first = "010"
            second = str(random.randint(1000, 9999))
            third = str(random.randint(1000, 9999))
            phone_numbers.append(f"{first}-{second}-{third}")
            phone_numbers.append(f"{first}{second}{third}")
        return phone_numbers

# 주민등록번호 생성기
class JuminNumberGenerator:
    def __init__(self, count):
        self.count = count  # 생성할 주민등록번호의 개수

    def generate(self):
        # 한국에서 상용될만한 주민번호 생성
        jumin_numbers = []
        for _ in range(self.count):
            # 생년월일 (YYMMDD)
            year = random.randint(1950, 2023) 
            month = random.randint(1, 12) 
            day = random.randint(1, 31)  

            try:
                birth_date = datetime(year, month, day)
                birth_date_str = birth_date.strftime("%y%m%d")  # YYMMDD 형식으로 변환
            except ValueError:
                continue  # 유효하지 않은 날짜일 경우 continue

            # 성별을 구분하는 첫 자리는 임의로 결정 (1, 2, 3, 4 중에서 랜덤으로)
            gender_digit = random.choice([1, 2, 3, 4])

            # 고유번호 (XXXXXX)
            unique_number = random.randint(100000, 999999)

            # 주민등록번호 생성 (YYMMDD-XXXXXXX)
            jumin_number = f"{birth_date_str}-{gender_digit}{unique_number}"
            jumin_numbers.append(jumin_number)
        
        return jumin_numbers

# 데이터 저장기
class DataSaver:
    def __init__(self, folder):
        self.folder = folder  # 데이터를 저장할 폴더

    # 데이터를 파일로 저장
    def save(self, file_name, data):
        with open(f"{self.folder}/{file_name}", "w") as file:
            for item in data:
                file.write(item + "\n")

#  데이터  생성기
class DataGenerator:
    def __init__(self, count):
        self.count = count
        self.phone_generator = PhoneNumberGenerator(count)
        self.jumin_generator = JuminNumberGenerator(count)
        self.saver = DataSaver('raw')

    def generate_phone_numbers(self):
        return self.phone_generator.generate()

    def generate_jumin_numbers(self):
        return self.jumin_generator.generate()

    def save_data(self, phone_numbers, jumin_numbers):
        self.saver.save('phone_numbers.txt', phone_numbers)
        self.saver.save('jumin_numbers.txt', jumin_numbers)
        print(f"Data successfully saved in 'data/raw' folder")

# 데이터 생성 및 저장 실행
data_generator = DataGenerator(50000)
phone_numbers = data_generator.generate_phone_numbers()
jumin_numbers = data_generator.generate_jumin_numbers()
data_generator.save_data(phone_numbers, jumin_numbers)