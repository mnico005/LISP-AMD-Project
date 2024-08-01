import vai_q_onnx

vai_q_onnx.quantize_static(
   "lisp2.ONNX",
   "lispQuant.ONNX",
   None,
   quant_format=vai_q_onnx.QuantFormat.QDQ,
   calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
   activation_type=vai_q_onnx.QuantType.QInt8,
   weight_type=vai_q_onnx.QuantType.QInt8,
)
