import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date

path = "face_database"
images = []
# 将数据库中的所有人名存储下来
person_name = []
# 将人脸数据库中的照片item以列表形式返回
my_list = os.listdir(path)
print(my_list)
# 将images, person_name初始化
for image in my_list:
    img = cv2.imread(f'{path}/{image}')
    images.append(img)
    person_name.append(os.path.splitext(image)[0])

print(person_name)


def find_encodings(images):
    """
    将数据库中的所有人脸照片进行编码
    :param images: 数据库中所有的照片list
    :return: 对所有照片进行编码生成一个编码list
    """
    encodings_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings_list.append(encode)

    return encodings_list


def register_info(name):
    """
    将识别到的人脸信息记录在文档中
    :param name:
    :return:
    """
    today = date.today()
    today_str = today.strftime("%Y%m%d")
    file_name = f'{today_str}_register_log.csv'
    # 创建当前打卡文件
    try:
        with open(file_name, 'x') as f:
            f.writelines('Name, Datetime')
    except FileExistsError:
        pass

    with open(file_name, 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        # 可以设置只登记在数据库人员or陌生人也登记
        if (name not in name_list) or (name in name_list):
            now = datetime.now()
            datetime_str = now.strftime("%Y/%d/%m, %H:%M:%S")
            f.writelines(f'\n{name},{datetime_str}')


# 调用find_encodings() 函数，得到编码list
encode_list_known = find_encodings(images)
print('Encoding completed.')

# 从摄像头获取图片
cap = cv2.VideoCapture(0)

# 在摄像头active的过程中持续进行人脸识别任务
while True:
    ret, frame = cap.read()
    # 对捕获的frame进行缩小
    imgSmall = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # 获得人脸位置坐标
    faceCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encode_list_known, encodeFace)
        face_distances = face_recognition.face_distance(encode_list_known, encodeFace)
        print(face_distances)
        best_match_index = np.argmin(face_distances)

        # 根据face distance来判断是不是数据库中存在的人，如果不是，登记其名字为Unknown
        if face_distances[best_match_index] < 0.50:
            name = person_name[best_match_index].upper()
            register_info(name)
        else:
            name = 'Unknown'
            register_info(name)
            # print(name)
        # 使用框框将脸部标注
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)


    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)
