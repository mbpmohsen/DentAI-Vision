# Tooth recognizer from radiology

### Raw Radiology Image:
![model response](./raw-image.png)
### Response:
![model response](./image.png)


### Scripts: 
Model has been trained in 'runs' directory but if you like to do it again, run this command:

```shell
python3 train_tooth_count.py
```
Run Application:

```shell
 uvicorn train_tooth_count_app:train_tooth_count_app --reload
```

> Project will serve in http://127.0.0.1:8000

### Test: 

```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_your_image/image.png'
```