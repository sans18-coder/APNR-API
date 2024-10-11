from flask import Flask, jsonify, request
from http import HTTPStatus
from transformers import VisionEncoderDecoderModel,TrOCRProcessor
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
# import os
import cv2

app = Flask(__name__)
app.config['MODEL_OBJECT_DETECTION']='./model/detect_plat.pt'
# app.config['RESULT_CROPED']='./crop'
app.config['ALLOWED_EXTENSIONS'] = set(['png','jpg','jpeg'])
model_detect = YOLO(app.config['MODEL_OBJECT_DETECTION'])
detect_names = model_detect.names

processor_ocr = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
model_ocr = VisionEncoderDecoderModel.from_pretrained("Sans1807/APNR-Braincore-V2")

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

def crop(image_path):
    
    # original_filename, ext = os.path.splitext(os.path.basename(image_path))

    im0 = cv2.imread(image_path)
    if im0 is None:
        raise ValueError(f"Error reading image file {image_path}")

    results = model_detect.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=detect_names)

    idx = 0
#   cropped_files = []
    if boxes is not None:
        for box, cls in zip(boxes, clss):
            idx += 1
            class_name = detect_names[int(cls)] 
            annotator.box_label(box, color=colors(int(cls), True), label=class_name)
            crop_obj = im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            #   crop_filename = os.path.join(app.config['RESULT_CROPED'], f"{original_filename}{idx}{ext}")
            #   cv2.imwrite(crop_filename, crop_obj)
            #   cropped_files.append(crop_filename)
    return crop_obj


def ocr(image):

    pixel_values = processor_ocr(image, return_tensors='pt').pixel_values
    generated_ids = model_ocr.generate(pixel_values)
    generated_text = processor_ocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

@app.route("/")
def index():
    return jsonify({
        "status" : {
            "code" : HTTPStatus.OK,
            "message" : "nyambung cuy santui",
        },
        "data" : None
    }),HTTPStatus.OK

@app.route("/prediction",methods=["POST"]) 
def predict():
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            crop_image = crop(reqImage)
            text = ocr(crop_image)
            return jsonify({
                            'status': {
                                'code': HTTPStatus.OK,
                                'message': 'Success predicting',
                            },
                            'data': text,
                        }),HTTPStatus.OK,
                
        else:
            return jsonify({
                            'status': {
                                'code': HTTPStatus.BAD_REQUEST,
                                'message': 'Invalid file format. Please upload a JPG, PNG, or JPEG image',
                            }
                            }),HTTPStatus.BAD_REQUEST,
                
    else:
        return jsonify({
                        'status': {
                            'code': HTTPStatus.METHOD_NOT_ALLOWED,
                            'message': 'Methode not allowed',
                        }
                        }),HTTPStatus.METHOD_NOT_ALLOWED,


if __name__ == '__main__': 
    app.run(
        host='0.0.0.0', port='5000', debug=True
    )
