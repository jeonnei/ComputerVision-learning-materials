import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('../img/sudoku.jpg')
img2 = img.copy()

def onChange(x):
    # 트랙바에서 값 가져오기
    rho = cv2.getTrackbarPos('Rho', 'Probabilistic Hough Lines')
    theta = cv2.getTrackbarPos('Theta', 'Probabilistic Hough Lines') / 100.0
    threshold = cv2.getTrackbarPos('Threshold', 'Probabilistic Hough Lines')
    minLineLength = cv2.getTrackbarPos('Min Line Length', 'Probabilistic Hough Lines')
    maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Probabilistic Hough Lines')

    # 그레이 스케일로 변환 및 엣지 검출
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray, 50, 200)

    # 확률적 허프 변환 적용
    lines = cv2.HoughLinesP(edges, rho, theta * np.pi / 180, threshold, None, minLineLength, maxLineGap)

    # 결과 이미지 초기화
    img2 = img.copy()

    # 검출된 선 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img2, (x1,y1), (x2, y2), (0,255,0), 2)

    # 원본 이미지와 결과 이미지를 나란히 표시
    cv2.imshow('Probabilistic Hough Lines', np.hstack((img, img2)))

# 'Probabilistic Hough Lines'라는 이름의 윈도우 생성
cv2.namedWindow('Probabilistic Hough Lines')

# 트랙바 생성
cv2.createTrackbar('Rho', 'Probabilistic Hough Lines', 1, 10, onChange)
cv2.createTrackbar('Theta', 'Probabilistic Hough Lines', 180, 360, onChange)
cv2.createTrackbar('Threshold', 'Probabilistic Hough Lines', 100, 200, onChange)
cv2.createTrackbar('Min Line Length', 'Probabilistic Hough Lines', 50, 200, onChange)
cv2.createTrackbar('Max Line Gap', 'Probabilistic Hough Lines', 10, 100, onChange)

# 초기 화면 출력을 위해 onChange 함수 호출
onChange(0)

# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()