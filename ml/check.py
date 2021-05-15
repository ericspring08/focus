import final_file
import cv2
final_model = final_file.get_model("hackethernet.h5")
img = cv2.imread("side.jpeg")
output = final_model.predict(img)
print(output)
