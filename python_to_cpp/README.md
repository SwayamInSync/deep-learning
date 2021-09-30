# Follow the below instructions to build C++ file

- Download the latest version of `libtorch` from [here](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip) then extract in the `build` folder.

- Inside the `build` directory
  
- create a `main.cpp` file and copy-paste the code given below

  ```c++
  #include <torch/script.h> // One-stop header.
  
  #include <iostream>
  #include <memory>
  
  int main(int argc, const char* argv[]) {
    if (argc != 2) {
      std::cerr << "usage: example-app <path-to-exported-script-module>\n";
      return -1;
    }
  
  
    torch::jit::script::Module module;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      return -1;
    }
  
    std::cout << "ok\n";
  }
  ```

- Create a `CMakeLists.txt` file and copy-paste the below text

  ```bash
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(custom_ops)
  
  find_package(Torch REQUIRED)
  
  add_executable(example-app example-app.cpp)
  target_link_libraries(example-app "${TORCH_LIBRARIES}")
  set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
  ```

- open terminal inside `build`

- ```bash
  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .
  ```

- ```bash
  cmake --build . --config Release
  ```

- Running:

- ```bash
  ./example-app <path_to_model>/traced_resnet_model.pt
  ok # get the output if everything works
  ```