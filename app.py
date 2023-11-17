import cv2
import json
import numpy as np
import ocr
import time
import cnn_detect
from flask import Flask, request, jsonify
from flask import Flask, render_template, request, flash
from PIL import Image
from flask import Flask, render_template, request, flash
from keras.preprocessing import image

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/ocr', methods = ['POST'])
def upload_file():
    start_time = time.time()

    if 'image' not in request.files:
        finish_time = time.time() - start_time

        return jsonify({
            'error': True,
            'message': "Foto wajib ada"
        })
    else:
        try:
            imagefile = request.files['image'].read()
            npimg = np.frombuffer(imagefile, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            fileimage = request.files['image'].stream
            fileimage = Image.open(fileimage)
            isimagektp = cnn_detect.main(fileimage)

            if isimagektp:
                (nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin, agama,
                status_perkawinan, provinsi, kabupaten, alamat, rt_rw, 
                kel_desa, kecamatan, pekerjaan, kewarganegaraan) = ocr.main(image)

                finish_time = time.time() - start_time

                if not nik or not nama or not provinsi or not kabupaten:
                    return jsonify({
                        'error': True,
                        'message': 'Resolusi foto terlalu rendah, silakan coba lagi.'
                    })

                return jsonify({
                    'error': False,
                    'message': 'Proses OCR Berhasil',
                    'result': {
                        'nik': str(nik),
                        'nama': str(nama),
                        'tempat_lahir': str(tempat_lahir),
                        'tgl_lahir': str(tgl_lahir),
                        'jenis_kelamin': str(jenis_kelamin),
                        'agama': str(agama),
                        'status_perkawinan': str(status_perkawinan),
                        'pekerjaan': str(pekerjaan),
                        'kewarganegaraan': str(kewarganegaraan),
                        'alamat': {
                            'name': str(alamat),
                            'rt_rw': str(rt_rw),
                            'kel_desa': str(kel_desa),
                            'kecamatan': str(kecamatan),
                            'kabupaten': str(kabupaten),
                            'provinsi': str(provinsi)
                        },
                        'time_elapsed': str(round(finish_time, 3))
                    }
                })
            else:
                return jsonify({
                    'error': True,
                    'message': 'Foto yang diunggah haruslah foto E-KTP'
                })
        except Exception as e:
            print(e)
            return jsonify({
                'error': True,
                'message': 'Maaf, KTP tidak terdeteksi'
            })

if __name__ == "__main__":
    app.run(debug = True)
