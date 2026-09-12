# ReazonSubtitle

영상 파일의 첫 번째 **원본 오디오 스트림만** FFmpeg로 디코딩한 뒤, Silero VAD로 음성 구간과 절대 타임스탬프를 찾고 일본어 STT로 인식합니다. 인식 결과를 로컬 HyTrans 모델로 한국어 번역하여 UTF-8 BOM 형식의 SRT로 저장하는 Windows GUI 프로그램입니다.

## 처리 방식

1. 영상은 건드리지 않고 `0:a:0` 오디오 스트림만 16 kHz mono PCM 분석본으로 추출합니다.
2. MekiAudioCapture의 `FAST` 기준을 적용합니다.
   - threshold 0.50
   - 최소 음성 0.20초 / 최소 무음 0.25초
   - 최대 구간 20초
   - 앞 0.15초 / 뒤 0.35초 패딩
   - 강제 분할 문맥 overlap 0.40초
3. VAD가 반환한 16 kHz 절대 샘플 위치를 유지하여 SRT 밀리초로 변환합니다. 강제 분할 overlap은 인식 문맥에만 쓰고 SRT 시작점에서는 제거합니다.
4. 기본 일본어 STT인 `sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8` 또는 옵션인 ReazonSpeech로 인식한 뒤, HY-MT1.5(기본) 또는 Hy-MT2(옵션) 로컬 ONNX 모델로 한국어 번역합니다.
5. 결과를 영상 옆의 `*.ko.srt`로 저장합니다.

## 일본어 STT 모델

- 기본값: Parakeet TDT-CTC 0.6B 일본어 INT8. Sherpa-ONNX의 NeMo CTC 모델이며 `model.int8.onnx`와 `tokens.txt`를 사용합니다.
- 옵션: ReazonSpeech 일본어. 기존 transducer 경로를 유지하며 INT8 또는 FP32를 선택할 수 있습니다.

Parakeet은 이름에 TDT가 포함되어도 ReazonSpeech의 `encoder + decoder + joiner` 형식이 아닙니다. 앱은 NeMo CTC 전용 recognizer로 별도 연결하므로, Python·PyTorch·NeMo 설치 없이 배포 EXE에서 실행됩니다.

## 실행

빌드 결과의 `dist\ReazonSubtitle\ReazonSubtitle.exe`를 실행합니다. Python 설치나 인터넷 연결은 필요하지 않습니다. HyTrans ONNX worker를 구동하기 위해 Windows 기본 Microsoft Edge 또는 Google Chrome 중 하나는 설치되어 있어야 합니다.

모델과 FFmpeg가 크므로 EXE만 따로 옮기지 말고 `ReazonSubtitle` 폴더 전체를 함께 옮겨야 합니다. 특히 `assets\sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8`와 `assets\reazonspeech-ja`를 모두 포함해야 두 STT 옵션이 작동합니다.

## 빌드

PowerShell에서 다음을 실행합니다.

```powershell
.\build.ps1
```

빌드는 Python 3.12 환경과 `requirements-build.txt`의 패키지를 사용하지만, 완성된 EXE에는 Python 런타임이 포함됩니다. 이 저장소와 같은 상위 폴더의 `MekiCopy\.build-python`이 있으면 그 재현 가능한 빌드 환경을 자동으로 사용합니다. 빌드 스크립트는 두 STT 모델 자산을 확인하고, 완성된 EXE에서 기본 Parakeet과 옵션 ReazonSpeech를 모두 로드하는 self-test를 실행합니다.

## 테스트

```powershell
C:\path\to\python.exe -m unittest discover -s .\tests -v
```

실제 1.3GB Hy-MT2 모델 로드와 한 문장 번역까지 확인하려면 다음 환경 변수를 추가합니다.

```powershell
$env:REAZON_RUN_TRANSLATION_SMOKE = "1"
C:\path\to\python.exe -m unittest tests.test_smoke.LocalAssetSmokeTests.test_hytrans_local_model -v
```
