import torch
import os
import cv2
import numpy as np
import sys
from mmdet.apis import init_detector, inference_detector
from yolox.tracker.byte_tracker import BYTETracker
from mmpretrain.apis import ImageClassificationInferencer
from yolox.utils.visualize import plot_tracking
import urllib.request

HOME = os.getcwd()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CAR_DETECTOR_CONFIG_PATH = f"{HOME}/configs/yolov8/car_detection.py"
CAR_DETECTOR_WEIGHTS_PATH = f"{HOME}/configs/yolov8/car_detection.pth"
CAR_CLASSIFIER_CONFIG_PATH = f"{HOME}/configs/yolov8/car_classifier.py"
CAR_CLASSIFIER_WEIGHT_PATH = f"{HOME}/configs/yolov8/car_classifier.pth"
LIGHT_DETECTOR_CONFIG_PATH = f"{HOME}/configs/yolov8/light_detection.py"
LIGHT_DETECTOR_WEIGHTS_PATH = f"{HOME}/configs/yolov8/light_detection.pth"
LIGHT_CLASSIFIER_CONFIG_PATH = f"{HOME}/configs/yolov8/light_classifier.py"
LIGHT_CLASSIFIER_WEIGHTS_PATH = f"{HOME}/configs/yolov8/light_classifier.pth"

# функции imcrop и pad_img_to_fit_bbox для правилной обрезки изображения
def imcrop(img, bbox):
   x1, y1, x2, y2 = bbox
   if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
   return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

# проверка есть ли на k фрэймах до last_ind информация о машине с индексом tid
def check_presence(results, tid, last_ind, k):
    flag = True
    if last_ind >= len(results) or k > last_ind:
        flag = False
    for i in range(1, k+1):
        if tid not in results[last_ind - i][1]:
            flag = False
            break
    return flag

# все модельки
car_detector_model = init_detector(CAR_DETECTOR_CONFIG_PATH, CAR_DETECTOR_WEIGHTS_PATH, device=DEVICE)
car_classifier_model = ImageClassificationInferencer(CAR_CLASSIFIER_CONFIG_PATH, CAR_CLASSIFIER_WEIGHT_PATH, device=DEVICE)
light_detector_model = init_detector(LIGHT_DETECTOR_CONFIG_PATH, LIGHT_DETECTOR_WEIGHTS_PATH, device=DEVICE)
light_classifier_model = ImageClassificationInferencer(LIGHT_CLASSIFIER_CONFIG_PATH, LIGHT_CLASSIFIER_WEIGHTS_PATH, device=DEVICE)

classes = {'car': (255, 0, 0), 'bus': (0, 255, 0), 'truck': (0, 0, 255)}


class ByteTrackArgument:
    track_thresh = 0.4  # High_threshold
    track_buffer = 40  # Number of frame lost tracklets are kept
    match_thresh = 0.65  # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0  # Minimum bounding box aspect ratio
    min_box_area = 1.0  # Minimum bounding box area
    mot20 = False  # If used, bounding boxes are not clipped.
    save_result = True



hls_url = "https://s2.moidom-stream.ru/s/public/0000001281.m3u8"

# Подключение к видео
cap = cv2.VideoCapture(hls_url)

# Проверка успешного открытия видео
if not cap.isOpened():
    print('Не удалось открыть видео.')
    exit()
    
# # считываем видео
# name_video="video.mp4"
# path_to_video = f"{HOME}/{name_video}"
# cap = cv2.VideoCapture(path_to_video)
img_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
img_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

tracker = BYTETracker(ByteTrackArgument)
frame_id = 0

results = []

car_images = {} # будет сохранять для каждого id машины последнее её изображение
max_bboxes = {}
counter = 0

# # подготавливаем новое видео для записи
# vid_writer = cv2.VideoWriter(f"{HOME}/result/full.avi", cv2.VideoWriter_fourcc(*"mp4v"), 25,
#                              (int(img_width), int(img_height)))


while True:
    # считываем видео по одному фрэйму, frame - одно изображение
    count = 0
    max_bboxes = {}
    ret, frame = cap.read()
    if not ret:
        break
    frame_id = frame_id + 1
    # детектор машины
    predict = inference_detector(car_detector_model, frame)

    car_detector_scores = predict.pred_instances.scores
    car_detector_bboxes = predict.pred_instances.bboxes
    # откидываем все машины с низким предсказанием
    index = 0
    for i in range(0, len(car_detector_scores)):
        if car_detector_scores[i] < 0.3:
            index = i
            break

    car_detector_bboxes = car_detector_bboxes[:index]
    car_detector_scores = car_detector_scores[:index]

    # изменяем размер, чтобы он подходил для ByteTrack 
    car_detector_scores = car_detector_scores.unsqueeze(1)
    concatination = torch.cat((car_detector_bboxes, car_detector_scores), dim=1)
    # запускаем трекер
    online_targets = tracker.update(concatination.cpu(), [img_height, img_width], [img_height, img_width])
    online_tlwhs = []
    online_ids = []
    online_scores = []
    check_size = []
    online_status=[]
    cars_info = {} # словарь вида car_id : [tlwh, score, status]
    # перебираем все найденные машины
    for t in online_targets:
        tlwh = t.tlwh  # bbox для всех найденных объектов в виду (x,y,width,height)
        tid = t.track_id  # id объекта
        
        x1, y1, w, h = tlwh
        x1, y1, w, h = map(int, (x1, y1, w, h))
        if w * h <= 1024:
            continue
        x2 = x1 + w
        y2 = y1 + h
        # вырезаем машину
        cropped = imcrop(frame, (x1, y1, x2, y2))
        # не отрисовываем класс автомобиля
        cropped = cropped.copy()
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        predict_car = car_classifier_model(cropped)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        # запускаем классификатор
        car_class = predict_car[0]['pred_class']
        car_clases = [] # сохраняем здесь 3 предсказания: на этом фрейме, на прошлом и на позапрошлом
        # выбираем из 3-х предсказаний самый часто встречающийся
        if (frame_id >= 4) and check_presence(results, tid, frame_id - 1, 2):
            car_clases.append(car_class)
            car_clases.append(results[frame_id - 2][1][tid][3])
            car_clases.append(results[frame_id - 3][1][tid][3])
            # выбираем самый часто встречающийся класс
            car_class = max(car_clases, key=car_clases.count)
        # отрисовываем bbox и подписываем класс объекта
        start_point = (x1, y1)
        end_point = (x2, y2)
        color = classes[car_class]
        thickness = 3
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 
        cv2.putText(img=frame, text=f'{car_class}', org=(x1, y1 - 5), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),thickness=1)

        # дальше идет запись всех результатов с машинами для дальнейшего использования
        vertical = tlwh[2] / tlwh[3] > 1.6
        tstatus = 0  # двигается ли машина на камеру или нет; default 0 - на камеру; 1 - против камеры
        if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
            last_ind = -1
            for i in range(len(results)):
                if tid in results[i][1]:
                    last_ind = i
                    last_car_info = results[i][1][tid]
            if last_ind != -1 and last_car_info[0][2] * last_car_info[0][3] > tlwh[2] * tlwh[3]:
                tstatus = 1
                cars_info[tid] = [tlwh, t.score, tstatus, car_class]

                # проверяем, что если на последних 5 фрэймах эта машина двигалась на камеру(с возможностью ошибки на одном фрейме)
                if (last_ind >= 7) and check_presence(results, tid, last_ind, 5):
                     if results[last_ind - 1][1][tid][2] + results[last_ind - 2][1][tid][2] + results[last_ind - 3][1][tid][2] + results[last_ind - 4][1][tid][2] + results[last_ind - 5][1][tid][2] <= 1:
                        # таким образом у нас получается, что до этого фрэйма машина двигалась на камеру, а теперь против неё, то есть начала выходить из кадра
                        if (tid in car_images):
                            # если есть, то добавляем предыдущее изображение
                            max_bboxes[count] = {'bbox': tlwh, 'image': car_images[tid], 'index': tid}
                        else: 
                            max_bboxes[count] = {'bbox': tlwh, 'image': cropped, 'index': tid}

                        count += 1
            if last_ind == -1:
                cars_info[tid] = [tlwh, t.score, tstatus, car_class]
            if last_ind != -1 and last_car_info[0][2] * last_car_info[0][3] < tlwh[2] * tlwh[3]:
                tstatus = 0
                cars_info[tid] = [tlwh, t.score, tstatus, car_class]
            if last_ind != -1 and last_car_info[0][2] * last_car_info[0][3] == tlwh[2] * tlwh[3]:
                tstatus = 0
                cars_info[tid] = [tlwh, t.score, tstatus, car_class]
            
            car_images[tid] = cropped


    results.append((frame_id, cars_info))  # добавляем всю информацию о сцене

    # рассматриваем каждый максимальный вырезанный автомобиль, проверяя что он достаточно большой
    for tid, data in max_bboxes.items():
        if data['image'] is not None and np.any(data['image']):
            if data['image'].shape[0] * data['image'].shape[1] >= 1024:
                # детектор фар
                predict_light = inference_detector(light_detector_model, data['image'])
                light_detector_scores = predict_light.pred_instances.scores
                light_detector_bboxes = predict_light.pred_instances.bboxes
                index = 0
                for i in range(0, len(light_detector_scores)):
                    if light_detector_scores[i] < 0.25:
                        index = i
                        break

                light_detector_bboxes = light_detector_bboxes[:index]
                light_detector_scores = light_detector_scores[:index]

                # Вырежем каждую фару по bbox и сохраним изображение
                for bbox_idx, bbox in enumerate(light_detector_bboxes):
                    x1, y1, x2, y2 = map(int, bbox)
                    light_image = imcrop(data['image'], (x1, y1, x2, y2))
                    # запускаем классификатор фар
                    sys.stdout = open(os.devnull, "w")
                    sys.stderr = open(os.devnull, "w")
                    predict_light = light_classifier_model(light_image)
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    light_class = predict_light[0]['pred_class']
                    # отрисовываем bboxs и классы на frame и на вырезанном изображении
                    add_text_image = data['image']

                    add_text_image = cv2.rectangle(add_text_image, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=1)
                    cv2.putText(img=add_text_image, text=f'{light_class}', org=(x1, y1 - 3), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
                    frame = cv2.rectangle(frame, (int(data['bbox'][0] + x1), int(data['bbox'][1] + y1)), (int(data['bbox'][0] + x2), int(data['bbox'][1] + y2)), color=(0, 0, 0), thickness=1)
                    cv2.putText(img=frame, text=f'{light_class}', org=(int(data['bbox'][0] + x1), int(data['bbox'][1] + y1 - 3)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=1)
                # сохраняем для каждой машины
                if (len(light_detector_bboxes) != 0):
                    cv2.imwrite(f"{HOME}/result/headlights/{data['index']}_light.jpg", add_text_image)


    cv2.imshow("Frame", frame)

    # Прерываем цикл по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # # записываем наш frame
    # vid_writer.write(frame)

cap.release()
cv2.destroyAllWindows()

