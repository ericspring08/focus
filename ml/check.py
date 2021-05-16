import final_file
import cv2
img_path = "front.jpeg"
final_model = final_file.get_model("hackethernet.h5")
img = cv2.imread(img_path)
output = final_model.predict(img)
print(output)
