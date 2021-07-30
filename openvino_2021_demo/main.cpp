#include <inference_engine.hpp>
#include <opencv2\opencv.hpp>
#include <samples\ocv_common.hpp>
#include <samples\classification_results.h>
#include <iostream>
#include <fstream>

int main()
{
	// --------------------------- 1. Load Inference engine instance -------------------------------------
	std::cout << "Loading Inference Engine......" << std::endl;
	InferenceEngine::Core ie;
	
	// --------------------------- 2. Read a model in OpenVINO IR    -------------------------------------
	InferenceEngine::CNNNetwork network = ie.ReadNetwork("models/squeezenet1.1.xml");

	// --------------------------- 3. Prepare input and output blobs    ----------------------------------
	// input
	InferenceEngine::InputsDataMap inputsInfo(network.getInputsInfo());
	printf("\tInput Node: %d\n", inputsInfo.size());
	std::string input_name = inputsInfo.begin()->first;

	auto inputInfoItem = *inputsInfo.begin();
	inputInfoItem.second->setPrecision(InferenceEngine::Precision::U8);
	inputInfoItem.second->setLayout(InferenceEngine::Layout::NHWC);

	// output
	InferenceEngine::OutputsDataMap outputsInfo(network.getOutputsInfo());
	printf("\tOutput Node: %d\n", outputsInfo.size());
	InferenceEngine::DataPtr output_info = outputsInfo.begin()->second;
	std::string output_name = outputsInfo.begin()->first;

	output_info->setPrecision(InferenceEngine::Precision::FP32);

	// --------------------------- 4. Loading model to the device ------------------------------------------
	InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
	//std::map<std::string, std::string> config = { { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } };
	//InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU", config);

	// --------------------------- 5. Create infer request -------------------------------------------------
	InferenceEngine::InferRequest inferRequst = executable_network.CreateInferRequest();

	// --------------------------- 6. Prepare input        -------------------------------------------------
	std::cout << "Starting Process input......" << std::endl;
	std::string img_name = "assert/car.png";
	cv::Mat image = cv::imread(img_name);
	cv::resize(image, image, cv::Size(227, 227));
	InferenceEngine::Blob::Ptr imgBlob = wrapMat2Blob(image);
	inferRequst.SetBlob(input_name, imgBlob);

	// --------------------------- 7. Do inference        -------------------------------------------------
	std::cout << "Do Inference......" << std::endl;
	inferRequst.Infer();
	
	// --------------------------- 8. Process output        -----------------------------------------------
	InferenceEngine::Blob::Ptr output = inferRequst.GetBlob(output_name);
	std::string label_name = "models/squeezenet1.1.labels";
	std::ifstream in(label_name);
	std::vector<std::string> labels;
	std::string line;
	if (in)
	{
		while (std::getline(in,line))
		{
			labels.push_back(line);
		}
	}
	ClassificationResult classificationResult(output, {img_name}, 1, 5, labels);
	classificationResult.print();

	/*或者输出使用这段代码可以得到输出层的数据
	InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
	auto moutputholder = moutput->rmap();
	float * o = moutputholder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
	*/

	getchar();
	return 0;
}