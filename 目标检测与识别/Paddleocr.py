from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang="en")
img_path = "/home/sci/zyn/test/2000807-6-CD8-DX1.jpg"
result = ocr.ocr(img_path,cls=True)
for line in result:
    print(line)