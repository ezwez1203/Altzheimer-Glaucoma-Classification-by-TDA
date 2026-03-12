import os
import glob


class IOSTARDataset:
    def __init__(self, root_dir, mode='train'):
        """
        Bob 라이브러리 없이 IOSTAR 데이터셋을 로드합니다.

        Args:
            root_dir (str): 데이터셋 폴더 경로 (예: 'data/IOSTAR')
            mode (str): 'train' (앞 20장) 또는 'test' (뒤 10장)
        """
        self.root_dir = root_dir
        self.mode = mode

        # 이미지 파일 목록 가져오기 (이름순 정렬 필수!)
        # IOSTAR 원본 이미지 폴더명에 따라 'image' 또는 'original' 등을 확인해주세요.
        # 보통 IOSTAR는 'image/*.jpg' 또는 '*.tif' 형식을 따릅니다.
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'image', '*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, 'mask', '*.tif')))  # 예시 확장자

        # 데이터가 30장이 맞는지 확인
        if len(self.image_paths) == 0:
            print(f"Warning: {root_dir}에서 이미지를 찾을 수 없습니다.")

        # 논문/Bob 프로토콜에 따른 분할
        # Train: 0~19 (20장), Test: 20~29 (10장)
        if mode == 'train':
            self.image_paths = self.image_paths[:20]
            self.mask_paths = self.mask_paths[:20]
        elif mode == 'test':
            self.image_paths = self.image_paths[20:]
            self.mask_paths = self.mask_paths[20:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 여기에 이미지 로드 코드(cv2.imread 등)를 넣으면 됩니다.
        img_path = self.image_paths[idx]
        return img_path


# 사용 예시
if __name__ == "__main__":
    # 경로만 실제 경로로 바꿔주세요!
    data_path = "/home/lucius/바탕화면/RetiQ_VCP/archive/IOSTAR"

    train_set = IOSTARDataset(data_path, mode='train')
    test_set = IOSTARDataset(data_path, mode='test')

    print(f"Train 개수: {len(train_set)}")  # 20이 나와야 함
    print(f"Test 개수: {len(test_set)}")  # 10이 나와야 함