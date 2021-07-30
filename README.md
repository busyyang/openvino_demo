# openvino_demo
安装了OpenVINO 2021.03版本，并完成了如下的设置，得到的VS 2015的项目，如果配置项目不一样可能需要按照如下修改一些信息(尤其版本带来的差异)。
1. 新建项目，`视图`->`其他窗口`->`属性管理器`，打开`Release | x64`的`Microsoft.Cpp.x64.user`，在`C/C++`->`常规`->`附加包含目录`中添加：
   ~~~
   C:\Program Files (x86)\Intel\openvino_2021\opencv\include
   C:\Program Files (x86)\Intel\openvino_2021\opencv\include\opencv2
   C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\inference_engine\include
   ~~~
2. 关闭`Microsoft.Cpp.x64.user`，在主窗口中打开`调试`->`<project_name> 属性...`：
    - Debug x64，在`链接器`->`常规`->`附加库目录`中添加：
      ~~~
      C:\Program Files (x86)\Intel\openvino_2021\opencv\lib
      C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\inference_engine\lib\intel64\Debug
      ~~~
    - Release x64，在`链接器`->`常规`->`附加库目录`中添加：
      ~~~
      C:\Program Files (x86)\Intel\openvino_2021\opencv\lib
      C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\inference_engine\lib\intel64\Release
      ~~~
3. 打开在`链接器`->`输入`->`附加依赖项`中添加
    - Debug x64
      ~~~
      opencv_calib3d451.lib
      opencv_core451.lib
      opencv_dnn451.lib
      opencv_features2d451.lib
      opencv_flann451.lib
      opencv_gapi451.lib
      opencv_highgui451.lib
      opencv_imgcodecs451.lib
      opencv_imgproc451.lib
      opencv_ml451.lib
      opencv_objdetect451.lib
      opencv_photo451.lib
      opencv_stitching451.lib
      opencv_video451.lib
      opencv_videoio451.lib
      inference_engine.lib
      inference_engine_c_api.lib
      inference_engine_ir_reader.lib
      inference_engine_legacy.lib
      inference_engine_lp_transformations.lib
      inference_engine_onnx_reader.lib
      inference_engine_preproc.lib
      inference_engine_transformations.lib
      ~~~
    - Release x64
      ~~~
      opencv_calib3d451d.lib
      opencv_core451d.lib
      opencv_dnn451d.lib
      opencv_features2d451d.lib
      opencv_flann451d.lib
      opencv_gapi451d.lib
      opencv_highgui451d.lib
      opencv_imgcodecs451d.lib
      opencv_imgproc451d.lib
      opencv_ml451d.lib
      opencv_objdetect451d.lib
      opencv_photo451d.lib
      opencv_stitching451d.lib
      opencv_video451d.lib
      opencv_videoio451d.lib
      inference_engined.lib
      inference_engine_c_apid.lib
      inference_engine_ir_readerd.lib
      inference_engine_legacyd.lib
      inference_engine_lp_transformationsd.lib
      inference_engine_onnx_readerd.lib
      inference_engine_preprocd.lib
      inference_engine_transformationsd.lib
      ~~~

4. 添加`opencv_xxx.dll`和`inference_engine_xxx.dll`的路径到path中，这样才能在运行的时候找到dll文件（写了path以后需要重启VS才能生效）。
   ~~~
   C:\Program Files (x86)\Intel\openvino_2021\openvino\opencv\bin
   C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\inference_engine\bin\intel64\Release
   C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\inference_engine\bin\intel64\Debug
   C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\inference_engine\external\tbb\bin
   C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\ngraph\lib
   ~~~

5. 编译成功，但是执行`Release`的时候出现`?fill@numa_topology@internal@tbb@@YAXPEAH@Z于动态链接库xxx/inference_engine.dll`上的错误时候，到编译好的可执行程序目录下，先执行以下`C:\Program Files(x86)\IntelSWTools\openvino\bin\setupvars.bat`，发现同窗口下执行文件就没有这个错了，应该是环境变量设置不完善的问题。用`Debug`模式调试没问题。
6. (optinal) 在`openvino_2021_demo`项目中，由于使用到了OpenVINO官方demo提供的一些help函数，所以添加了一个额外的include引用地址，如果没有适用到的话，可以不添加这个不是必须的：
   ~~~
   C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\inference_engine\samples\cpp\common
   ~~~

7. 用一个测试代码，能运行起来的话就安装没问题(Debug模式下)，里面主要的逻辑就是:加载模型，处理输入输出的blob，模型加载到设备上，创建推理请求，然后把数据喂到输入节点上，执行推理请求，数据后处理的过程。后三个步骤可以执行多次，前面的步骤执行一次就可以了。
    - 这里单独理解了一下`InferenceEngine::InputsDataMap`和`InferenceEngine::OutputsDataMap`这两个数据类型，主要是用来处理模型的输入输出节点的，打开可以看到都是`std::map<std::string, DataPtr>`这个类型的，第一个是`std::string`数列类型的，是节点的名字，第二个是`DataPtr`数据类型的，查看代码是叫一种`smart pointer to the xxx instance`。
    - 通过`inferRequst.SetBlob(input_name, imgBlob)`设置输入层的数据blob，这里调用了`samples/ocv_common.hpp`文件中的`wrapMat2Blob`函数将cvMat数据类型转化为`InferenceEngine::Blob::Ptr`类型进行输入。在输出的时候`InferenceEngine::Blob::Ptr output = inferRequst.GetBlob(output_name)`获取输出节点的数据。可以通过`InferenceEngine::SizeVector output_size = output->getTensorDesc().getDims();`来查看输入节点的维度。
    - 在输出的部分，直接调用了`samples\classification_results.h`文件中的`ClassificationResult`来处理的。如果不调用这个的话，可以使用如下代码得到output的数据：
      ~~~cpp
      InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
      auto moutputholder = moutput->rmap();
      float * o = moutputholder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
      ~~~

8. 参考这个网址配置环境：https://www.bilibili.com/video/BV1Hz4y1U7g6?t=1418
9.  参考这个网址写一个简单的demo：https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/samples/hello_classification/main.cpp
10. 模型的下载参考`C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\demo_squeezenet_download_convert_run.bat`下载的文件即可。
