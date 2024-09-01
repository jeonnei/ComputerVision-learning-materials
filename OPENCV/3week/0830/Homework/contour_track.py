import cv2
import numpy as np

# 영상 읽기
img = cv2.imread('../img/shapes_donut.png')
img2 = img.copy()

def onChange(x):
    # 트랙바에서 threshold 값 가져오기
    threshold = cv2.getTrackbarPos('Threshold', 'Contours')
    
    # 그레이스케일로 변환
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # threshold 값을 사용하여 이진화
    ret, imthres = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 모든 컨투어를 트리 계층으로 수집
    # RETR_TREE: 모든 컨투어와 계층 구조를 완전히 구성
    # CHAIN_APPROX_SIMPLE: 컨투어 라인을 직선으로 근사화하여 메모리 절약
    contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # 결과 이미지 초기화 (원본 이미지 복사)
    img2 = img.copy()
    
    # 모든 컨투어 그리기
    for idx, cont in enumerate(contour2):
        # 각 컨투어마다 랜덤한 색상 생성
        color = [int(i) for i in np.random.randint(0,255, 3)]
        
        # 컨투어 그리기
        # -1은 모든 컨투어를 그린다는 의미, 3은 선 두께
        cv2.drawContours(img2, contour2, idx, color, 3)
        
        # 컨투어의 시작점에 인덱스 번호 표시
        cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    
    # 원본 이미지와 컨투어가 그려진 이미지를 나란히 표시
    cv2.imshow('Contours', np.hstack((img, img2)))

# 'Contours'라는 이름의 윈도우 생성
cv2.namedWindow('Contours')

# 'Threshold'라는 이름의 트랙바 생성
# 초기값 127, 최대값 255로 설정
cv2.createTrackbar('Threshold', 'Contours', 127, 255, onChange)

# 초기 화면 출력을 위해 onChange 함수 호출
onChange(0)

# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()