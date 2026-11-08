"""Tầng phát hiện đầu (tuỳ chọn) cho chế độ realtime: DEIMv2-Wholebody49 (PINTO0309, Apache-2.0, ONNX) lớp Head (id 7)
chạy TRONG crop rider (xe + người, mở rộng), thay cho vùng đầu suy từ box người.
I/O theo demo của PINTO (demo/demo_hfhpe_onnx.py): resize thẳng về HxW cố định, BGR->RGB, /255 (HGNetV2) hoặc ImageNet (DINOv3),
đầu ra `label_xyxy_score` [1,Q,6] = (class, x1, y1, x2, y2, score), toạ độ chuẩn hoá nếu không có input orig_target_sizes."""
from __future__ import annotations
import time
import numpy as np, cv2
HEAD = 7
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

def onnx_gather_elements(data, indices, axis):
    """Lower ONNX GatherElements to indexed coordinates, preserving smaller non-axis
    dimensions (e.g. data[B,8400,4], indices[B,1240,1]). OpenVINO's native
    GatherElements validator requires equal non-axis dimensions for this graph.
    GatherND-8 in OpenVINO>=2025.3 also preserves ONNX negative-index semantics.
    """
    from openvino import opset13 as ops
    rank = data.get_partial_shape().rank.get_length()
    if not -rank <= axis < rank: raise ValueError("GatherElements axis out of range")
    axis %= rank
    indices = ops.convert(indices, "i64")
    shape = ops.shape_of(indices, "i64")
    coordinates = []
    for dimension in range(rank):
        if dimension == axis:
            coordinate = indices
        else:
            length = ops.gather(shape, ops.constant(dimension, np.int64), ops.constant(0, np.int64))
            values = ops.range(ops.constant(0, np.int64), length, ops.constant(1, np.int64), "i64")
            reshape = [1] * rank; reshape[dimension] = -1
            coordinate = ops.broadcast(ops.reshape(values, ops.constant(reshape, np.int64), False), shape)
        coordinates.append(ops.unsqueeze(coordinate, ops.constant(-1, np.int64)))
    return ops.gather_nd(data, ops.concat(coordinates, -1), batch_dims=0)


class DeimHeadDetector:
    def __init__(self, onnx_path, device="cuda", min_score=0.1, require_gpu=False,
                 threads=0, spinning=True, backend="onnx", openvino_device="CPU"):
        self.backend = backend
        if backend == "onnx":
            import onnxruntime as ort
            if device.startswith("cuda") and hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            available = ort.get_available_providers()
            if require_gpu and "CUDAExecutionProvider" not in available:
                raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
            cuda_index = int(device.split(":", 1)[1]) if ":" in device else 0
            prov = ([("CUDAExecutionProvider", {"device_id": cuda_index}), "CPUExecutionProvider"]
                    if device.startswith("cuda") else ["CPUExecutionProvider"])
            self.s = ort.InferenceSession(str(onnx_path), sess_options=options, providers=prov)
            self.providers = self.s.get_providers()
            if require_gpu and self.providers[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"Head detector fell back from CUDA: {self.providers}")
            i = self.s.get_inputs()[0]; self.inp = i.name; self.hw = (int(i.shape[2]), int(i.shape[3]))
            self.names = {x.name for x in self.s.get_inputs()}
        elif backend == "openvino":
            import openvino as ov
            from openvino.frontend import ConversionExtension
            import threading
            core = ov.Core()
            def convert_gather(node):
                return [onnx_gather_elements(node.get_input(0), node.get_input(1), node.get_attribute("axis", 0)).output(0)]
            core.add_extension(ConversionExtension("GatherElements", convert_gather))
            if openvino_device not in core.available_devices:
                raise RuntimeError(f"OpenVINO device {openvino_device} unavailable: {core.available_devices}")
            options = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
            if openvino_device == "CPU" and threads: options["INFERENCE_NUM_THREADS"] = threads
            model = core.read_model(str(onnx_path))
            self.compiled = core.compile_model(model, openvino_device, options)
            self.request = self.compiled.create_infer_request(); self._infer_lock = threading.Lock()
            self.inp = self.compiled.input(0).get_any_name()
            partial_shape = self.compiled.input(0).partial_shape
            self.hw = (partial_shape[2].get_length(), partial_shape[3].get_length())
            self.names = {p.get_any_name() for p in self.compiled.inputs}
            self._output = self.compiled.output("label_xyxy_score")
            self.providers = ["OpenVINO:" + str(d) for d in self.compiled.get_property("EXECUTION_DEVICES")]
        else:
            raise ValueError(f"Unsupported head runtime: {backend}")
        self.norm = "imagenet" if "dinov3" in str(onnx_path).lower() else "div255"
        self.min_score = min_score; self.t = 0.0; self.n = 0
    def __call__(self, img):
        """-> list (score, [x1,y1,x2,y2]) các đầu trong ảnh (pixel), sắp theo score giảm dần."""
        H, W = img.shape[:2]; h, w = self.hw
        x = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        if self.norm == "imagenet": x = (x - MEAN) / STD
        feed = {self.inp: x[None]}
        if "orig_target_sizes" in self.names: feed["orig_target_sizes"] = np.array([[W, H]], np.float32)
        t = time.perf_counter()
        if self.backend == "onnx":
            (p,) = self.s.run(["label_xyxy_score"], feed)
        else:
            with self._infer_lock: p = np.array(self.request.infer(feed)[self._output], copy=True)
        self.t += time.perf_counter() - t; self.n += 1
        p = p[0]; p = p[(p[:, 0].astype(int) == HEAD) & (p[:, 5] >= self.min_score)]
        out = []
        for _, x1, y1, x2, y2, sc in p:
            if "orig_target_sizes" not in self.names: x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
            if x2 - x1 >= 2 and y2 - y1 >= 2: out.append((float(sc), [float(x1), float(y1), float(x2), float(y2)]))
        return sorted(out, key=lambda z: -z[0])

def square(b, margin=0.05):
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1; x1 -= w * margin; x2 += w * margin; y1 -= h * margin; y2 += h * margin
    s = max(x2 - x1, y2 - y1); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]
