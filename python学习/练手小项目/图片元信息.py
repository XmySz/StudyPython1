import exifread


# 获取照片位置
class Image_Location():

    def __init__(self, image_path):
        self.img_path = image_path

    def lati_long_date(self, data):
        """
        对经纬度进行处理，保留６位小数
        """

        # 删除左右括号，最后以逗号分隔为一个列表
        data_list_tmp = str(data) .replace('[', '').replace(']', '').split(',')

        # 循环取出每个元素，删除元素两边的空格，得到一个新列表
        data_list = [date.strip() for date in data_list_tmp]

        # 替换秒的值
        data_tmp = data_list[-1].split('/')

        # 秒的值
        data_sec = int(data_tmp[0]) / int(data_tmp[1]) / 3600

        # 替换分的值
        data_tmp = data_list[-2]

        # 分的值
        data_minute = int(data_tmp) / 60

        # 度的值
        data_degree = int(data_list[0])

        # 由于高德API只能识别到小数点后的6位
        # 需要转换为浮点数，并保留为6位小数
        result = "%.6f" % (data_degree + data_minute + data_sec)
        return float(result)

    def get_image_exif(self):
        """
        获取图片的属性：纬度，经度，拍摄时间等
        """

        # 通过exifread获取图片的属性
        img_read = open(self.img_path, 'rb')
        img_exif = exifread.process_file(img_read)

        # 读取到的属性
        if img_exif:
            latitude = ""
            latitude_direction = ""
            longitude = ""
            take_time = ""
            take_equipment = ""
            take_model = ""
            longitude_direction = ""
            for exif in img_exif:

                # 纬度
                if exif == "GPS GPSLatitude":
                    latitude = img_exif["GPS GPSLatitude"]

                # 纬度的方向
                elif exif == "GPS GPSLatitudeRef":
                    latitude_direction = img_exif["GPS GPSLatitudeRef"]

                # 经度
                elif exif == "GPS GPSLongitude":
                    longitude = img_exif["GPS GPSLongitude"]

                # 经度方向
                elif exif == "GPS GPSLongitudeRef":
                    longitude_direction = img_exif["GPS GPSLongitudeRef"]

                # 拍摄时间
                elif exif == "EXIF DateTimeDigitized":
                    take_time = img_exif["EXIF DateTimeDigitized"]

                # 拍摄设备
                elif exif == "Image Make":
                    take_equipment = img_exif["Image Make"]

                # 拍摄型号
                elif exif == "Image Model":
                    take_model = img_exif["Image Model"]

            # 对获取的经纬度信息进一步处理
            if latitude and longitude:
                latitude_date = self.lati_long_date(latitude)
                longitude_date = self.lati_long_date(longitude)
                print("纬度为：", latitude_date)
                print("经度为：", longitude_date)
            print("拍摄时间为：", take_time)
            print("拍摄设备为: ", take_equipment)
            print("拍摄型号为：", take_model)

        else:
            print("图像信息为空，可能是上传的不是原图")


if __name__ == '__main__':
    location = Image_Location('p2.jpg')
    location.get_image_exif()
