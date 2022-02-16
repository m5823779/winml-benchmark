#include "pch.h"

using namespace winrt::Windows::Media;
using namespace winrt::Microsoft::AI::MachineLearning;

// get model path
winrt::hstring modelPath = L"./model.onnx";

int input_width = 352;
int input_height = 192;
int input_channel = 3;
int inference_time = 1000;

// windows machine learning
LearningModel model = nullptr;
LearningModelSession session = nullptr;
LearningModelDevice device = nullptr;

int total_time = 0;
int user_input;

void ParseArgs(int argc, char* argv[]) {
    // get onnx file path
    if (argc < 2)
        return;

    modelPath = winrt::hstring(wstring_to_utf8().from_bytes(argv[1]));
    if (argc >= 3)
        input_width = std::stoi(argv[2]);
    if (argc >= 4) 
        input_height = std::stoi(argv[3]);
    if (argc >= 5)
        inference_time = std::stoi(argv[4]);
}

// Entry
int main(int argc, char* argv[]) {
    // did they pass in the args 
    printf("Usage: %s [ ONNX file path ] [ Input width ] [ Input height ] [ Inference times ] \n", argv[0]);
    ParseArgs(argc, argv);

    // Input shape for model must be a multiple of 32
    if (input_width % 32 != 0) {
        input_width = input_width - (input_width % 32);
    }

    if (input_height % 32 != 0) {
        input_height = input_height - (input_height % 32);
    }

    // Check model exsit or not
    std::string s(modelPath.begin(), modelPath.end());
    struct stat buffer;
    bool exsit = stat(s.c_str(), &buffer) == 0;
    if (!exsit) {
        printf("Model not exsit ...");
        return 0;
    }

    // Select adapter
    printf("\nUse CPU (0) or GPU (1) to inference: ");
    std::cin >> user_input;
    printf("\n");

    if (user_input == 0) 
        device = LearningModelDevice(LearningModelDeviceKind::Cpu);
    else {
        D3D_FEATURE_LEVEL FeatureLevels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_9_1 };
        UINT NumFeatureLevels = ARRAYSIZE(FeatureLevels);
        D3D_FEATURE_LEVEL FeatureLevel;

        UINT i = 0;
        IDXGIAdapter* pAdapter;
        std::vector <IDXGIAdapter*> vAdapters;
        IDXGIFactory1* pFactory = NULL;
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);

        // Show all adapter
        while (pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND)
        {
            DXGI_ADAPTER_DESC adapter_desc;
            pAdapter->GetDesc(&adapter_desc);
            printf("Adapter %d: %ls\n", i, adapter_desc.Description);
            vAdapters.push_back(pAdapter);
            ++i;
        }

        printf("\nChoose adapter (0 - %d): ", i - 1);
        int user_input_adapter;
        std::cin >> user_input_adapter;

        int inference_adapter = user_input_adapter;

        // Create D3D11 Device
        winrt::com_ptr<ID3D11Device> m_inference_device;
        winrt::com_ptr<ID3D11DeviceContext> m_inference_context;
        D3D11CreateDevice(vAdapters[inference_adapter], D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, FeatureLevels, NumFeatureLevels, D3D11_SDK_VERSION, m_inference_device.put(), &FeatureLevel, m_inference_context.put());

        IDXGIDevice* m_DxgiDevice = nullptr;
        m_inference_device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&m_DxgiDevice));

        auto m_d3d_device = CreateDirect3DDevice(m_DxgiDevice);
        device = LearningModelDevice::CreateFromDirect3D11Device(m_d3d_device);
    }

    printf("\n");
    printf("Input width : %d\n", input_width);
    printf("Input height : %d\n", input_height);
    printf("Inference times : %d\n", inference_time);

    // Loading model
    model = LearningModel::LoadFromFilePath(modelPath);
    printf("Loading modelfile '%ws'\n", modelPath.c_str());
    printf("Load WINML Model [SUCCEEDED]\n");

    // Create a session and binding
    LearningModelSessionOptions sessionOptions;
    // Define input dimensions to concrete values in order to achieve better runtime performance
    sessionOptions.OverrideNamedDimension(L"input_cx", input_height);
    sessionOptions.OverrideNamedDimension(L"input_cy", input_width);
    sessionOptions.BatchSizeOverride(1);

    session = LearningModelSession(model, device, sessionOptions);
    LearningModelBinding binding(session);

    // Define input shape
    std::vector<int64_t> inputShape = { input_height, input_width, input_channel };

    // Create memory for input array
    float* pCPUInputTensor;
    uint32_t uInputCapacity;

    // Create WinML tensor float
    TensorFloat tf = TensorFloat::Create(inputShape);
    winrt::com_ptr<ITensorNative>  itn = tf.as<ITensorNative>();

    // Gets the tensor¡¦s buffer as an bytes array
    itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUInputTensor), &uInputCapacity);

    D3D11_MAPPED_SUBRESOURCE vfRes;
    ZeroMemory(&vfRes, sizeof(D3D11_MAPPED_SUBRESOURCE));

    float* tensorData = pCPUInputTensor;

    // bind the intput image
    // A list of the model's input features.
    auto&& description = model.InputFeatures().GetAt(0);

    // Create binding and then bind input features
    binding.Bind(description.Name(), tf);

    // run the model (warm up)
    printf("\nRunning the model...\n");
    auto results = session.Evaluate(binding, L"");


    for (int i = 0; i < inference_time; i++) {
        INT64 inference_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        auto results = session.Evaluate(binding, L"RunId");

        INT64 inference_end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        int timer = (int)(inference_end_time - inference_start_time);
        total_time += timer;
        printf("\rInference time : %d (ms)", timer);
    }

    printf("\n");
    printf("\nAverage inference time : %d (ms)", total_time / inference_time);
}

