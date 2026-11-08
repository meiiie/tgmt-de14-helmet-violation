# Workflow local / Colab / Kaggle — 2026-11-05

Mục tiêu: đo cùng pipeline trên CPU local và GPU cloud, giữ nguyên dữ liệu đánh giá và quyết định thay đổi dựa trên kết quả. Notebook mới là `notebooks/helmet_runtime_colab_kaggle.ipynb`; nó gọi CLI repository, không sao chép thuật toán inference vào notebook. Không có retraining hoặc thay backbone tự động.

**Trạng thái phiên này:** chưa thực thi GPU Colab/Kaggle. Không có thông tin xác thực Kaggle cloud trong môi trường hiện tại; không tạo tài nguyên trả phí. Việc compile notebook hoặc đóng gói ZIP không xác nhận CUDA hoạt động hay chứng minh FPS GPU. Kết quả Kaggle cũ trong `docs/results/kaggle_v9/` có cấu hình/probe riêng, không đại diện cho lần chạy mới.

## 1. Đóng gói code đang làm, không cần push Git

Tại thư mục repository trên local:

```powershell
python scripts/package_cloud_workspace.py
```

Output mặc định: `outputs/cloud/helmet_workspace.zip`. Archive lấy source hiện tại, bao gồm thay đổi chưa commit trong `src/`, `scripts/`, `configs/`, root `app.py`, requirements, README, notebook mới và tài liệu workflow liên quan. Tên thành viên ZIP dùng dấu `/`, có thứ tự và timestamp cố định. Không lấy `.git`, `.venv`, `.env`, data, outputs, cache Python, symlink, hay checkpoint model. Chỉ những phần source được chọn mới nằm trong package; đây không phải bản sao lưu toàn bộ repository.

Để mang theo probe nhỏ hiện có cùng provenance, dùng flag rõ ràng:

```powershell
python scripts/package_cloud_workspace.py --include-probe
```

Flag chỉ thêm `weights/dino_probe.joblib` và `weights/dino_probe_provenance.json`, mỗi file tối đa 4 MiB. Model YOLO, head ONNX, CLIP và DINO checkpoint không được gói. Probe serialize bằng joblib chỉ nên nạp từ nguồn bạn tin cậy. Notebook kiểm tra SHA256 probe theo provenance trước khi chạy. Có thể dùng `--out <path.zip>` để đổi nơi lưu.

Colab: upload notebook qua File → Upload notebook, bật GPU, rồi cell project sẽ nhận ZIP qua hộp upload. Kaggle: tạo notebook có GPU, thêm ZIP code dưới dạng **private dataset**, điền đường dẫn thật `/kaggle/input/<your-private-dataset>/helmet_workspace.zip` vào `PROJECT_ZIP`; code được giải nén vào `/kaggle/working/helmet_runtime`. Nếu đã có checkout đầy đủ trong thư mục ghi được, điền `PROJECT_DIR` và bỏ upload. Không có remote GitHub hardcode và không nhúng token vào URL clone.

## 2. Môi trường GPU và model

Notebook giữ phiên bản GPU PyTorch mà dịch vụ đã cài, ràng buộc torch/torchvision khi giải dependencies. Nó không cài requirements CPU local. Gỡ cả distribution ORT cũ rồi cài duy nhất `onnxruntime-gpu` đã chọn; restart Python trước bước tiếp theo để tránh module cũ còn trong memory.

Notebook cài thêm package telemetry tùy chọn `nvidia-ml-py` để benchmark lấy VRAM theo process qua NVML, bao gồm ONNX Runtime. Nếu NVML không hoạt động, metric phải ghi unavailable; Torch-only memory không đại diện cho tổng VRAM của pipeline có ORT.

Kiểm tra stack thực tế, không chỉ driver: Torch CUDA, cuDNN, GPU model, tổng VRAM, ORT version/providers. Theo [tài liệu CUDA Execution Provider chính thức](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), cần khớp CUDA/cuDNN với wheel; cuDNN 8 và 9 không thay thế nhau. Notebook dùng ORT 1.22.0 với CUDA 12.8+ và cuDNN 9; CUDA 12 thấp hơn với cuDNN 9 dùng 1.20.1. Stack khác dừng để chọn wheel theo bảng, không cài latest tùy ý. Import Torch trước ORT và preload thư viện khi API hỗ trợ. Provider được liệt kê chưa đủ chứng minh model chạy GPU; benchmark còn phải kiểm tra session thực tế.

Gated DINOv3 cần tài khoản đã được cấp quyền model. Lưu `HF_TOKEN` trong Colab Secrets hoặc Kaggle Secrets và cho notebook quyền đọc; không dán token vào source/cell/output. Có thể stage checkpoint riêng được cấp phép và đặt `LOCAL_DINO_PARENT` tới thư mục cha chứa `dinov3-vits16-pretrain-lvd1689m/`. Không công khai checkpoint/cache gated hoặc thông tin xác thực. Nếu probe provenance có phiên bản scikit-learn, notebook cài đúng phiên bản đó để hạn chế lỗi deserialization; lỗi compatibility phải được xử lý trước benchmark, không tự train probe thay thế.

`python scripts/prepare_accuracy_models.py` tải model head ONNX + CLIP dùng cho profile. YOLO loader tải checkpoint chuẩn nếu thiếu. Giữ cùng model/probe SHA256 khi so sánh; download và model load là chi phí startup riêng. Package ZIP không thể làm file local tự xuất hiện trên cloud: phải upload ZIP và artifact riêng, hoặc tải bằng script trong runtime.

## 3. Benchmark trước, chọn profile sau

Tạo mẫu held-out nhỏ, hoặc đưa **cùng** manifest và file frame/annotation của benchmark local lên runtime:

```bash
python scripts/prepare_benchmark_data.py --clips 4 --frames-per-clip 4
python scripts/benchmark_runtime.py --profile baseline_cuda --manifest data/benchmark/manifest.json --repeats 2 --out outputs/benchmarks/cloud_gpu_run/baseline_cuda
python scripts/benchmark_runtime.py --profile cloud_cuda --manifest data/benchmark/manifest.json --repeats 2 --out outputs/benchmarks/cloud_gpu_run/cloud_cuda
```

Hai profile CUDA chạy trên cùng GPU, cùng manifest và cùng device/thread setup. `baseline_cuda` tắt batch head/cache text, `cloud_cuda` bật hai tối ưu đó; đây là cặp so sánh ảnh hưởng tối ưu trên cùng phần cứng. Mỗi profile mặc định chạy 2 repeat; 16 frame vẫn chỉ là smoke benchmark. So CPU local với GPU cloud không tách riêng được tác động tối ưu khỏi khác biệt phần cứng.

Lệnh local dùng cùng CLI và `configs/runtime_profiles.json`:

```bash
python scripts/benchmark_runtime.py --profile baseline_cpu --manifest data/benchmark/manifest.json --out outputs/benchmarks/baseline_cpu
python scripts/benchmark_runtime.py --profile optimized_cpu --manifest data/benchmark/manifest.json --out outputs/benchmarks/optimized_cpu
python scripts/benchmark_runtime.py --profile openvino_cpu --manifest data/benchmark/manifest.json --out outputs/benchmarks/openvino_cpu
```

Mẫu 16 frame là lần kiểm tra đầu, không phải kết luận accuracy triển khai. Chỉ so sánh những run có cùng manifest/hash, nhãn, model/probe, ngưỡng, input size, precision và phạm vi metric. Nếu profile thay đổi precision, ghi rõ và kiểm tra cả sai lệch kết quả. Giữ startup/warmup riêng khỏi inference; đọc latency p50/p95, throughput và metrics accuracy/coverage cùng nhau. Record GPU model, Torch/ORT version, provider thực tế và VRAM; CUDA/profile bị lỗi phải báo lỗi, không ghi CPU fallback thành GPU.

Notebook export riêng thư mục `outputs/benchmarks/cloud_gpu_run` chứa cả `baseline_cuda/` và `cloud_cuda/` thành ZIP để download hoặc lưu private Kaggle output trước khi session hết hạn. Kết quả có thể chứa frame/prediction; xem nội dung trước khi chia sẻ. Giữ config, manifest, hashes và metric gốc cùng run; không chỉ chép một con số FPS. Không export toàn workspace hoặc HF cache.

## 4. Dữ liệu và training đề xuất sau khi đọc benchmark

Chưa khởi chạy huấn luyện trong workflow này. Quyết định tiếp theo cần dựa trên loại lỗi đo được:

1. Từ video thực có quyền sử dụng, thu thập hard cases ngoài HELMET: đầu nhỏ/xa, bị che, nhiều người trên một xe, trẻ em, mũ vải/cap/mũ lông/mũ chống nắng, trời tối, blur, góc camera cao và cảnh giao thông Việt Nam. Ghi clip/location/time cùng nguồn và quyền sử dụng.
2. Annotate head box thực, liên kết rider/xe khi có thể, nhãn `helmet`, `no_helmet`, `unknown/occluded`; phân biệt mũ thời trang với mũ bảo hiểm. Cho người thứ hai review các case mơ hồ. Không suy nhãn chính xác từ một crop không đủ chi tiết.
3. Chia train/validation/test theo **clip và location trước khi sinh crop**; frame gần nhau không rải vào nhiều split. Giữ nguyên official test clips và bộ test địa phương độc lập, không dùng chúng để fit probe, chọn threshold hoặc mine training. Kiểm tra trùng lặp giữa các nguồn.
4. Trước training, khóa baseline: head detection recall/IoU, crop precision/recall/F1/AUROC và coverage/unknown; full-system rider/track metrics theo protocol benchmark; breakdown riêng cho hats, trẻ em và small heads. Không đánh đồng crop F1 với track accuracy. Chốt tiêu chí accuracy, latency và tỷ lệ abstain trước khi xem test.
5. Nếu lỗi chủ yếu do crop/head sai, ưu tiên annotation/crop và kiểm định detector; nếu crop đúng nhưng phân loại sai mới cân nhắc train probe trên real training data. Chỉ thay thành phần đã có bằng chứng; không mặc định thêm tầng hoặc backbone lớn hơn.

Sau khi xác định thành phần gây lỗi và xác minh dữ liệu/nhãn training, tái sử dụng code đã có trong **workspace riêng để bảo vệ baseline**:

```bash
python scripts/prepare_probe_data.py --clips 40 --frames-per-clip 8
python src/build_head_crops.py
python src/train_probe.py facebook/dinov3-vits16-pretrain-lvd1689m
```

Các lệnh trên là quy trình mẫu hiện có cho official training clips, không phải đã chạy trong nhiệm vụ này. Với dữ liệu real mới, cần import annotation vào schema hiện có và kiểm tra split trước khi gọi bước crop/train; không đưa manifest held-out vào training. `src/train_probe.py` dùng GroupKFold theo clip và có thể ghi đè `weights/dino_probe.joblib`; giữ baseline ngoài workspace training và lưu provenance mới gồm model ID/revision, checksum, seed, manifest/split và package versions. Kiểm định candidate trên cùng test độc lập, kèm coverage và latency trước khi promote. Notebook không có cell training được chạy bởi Run all.

## Nguồn dịch vụ

[Colab FAQ](https://research.google.com/colaboratory/faq.html) nêu tài nguyên miễn phí không được bảo đảm và runtime có thể hết hạn; workflow dùng notebook tương tác, không dựng web service/tunnel/remote desktop. [Repository image chính thức của Kaggle](https://github.com/Kaggle/docker-python) là nguồn kiểm tra môi trường CPU/GPU; luôn ghi phiên bản runtime thực tế của lần chạy. Không hứa GPU cụ thể, quota cố định hoặc FPS trước khi đo.
