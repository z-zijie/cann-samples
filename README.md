# CANN Samples

## 快速开始

1. 配置项目

   使用以下命令初始化构建配置，CMake会自动创建`build`文件夹
   ```sh
   cmake -S . -B build
   ```

2. 执行编译

   使用CMake指令进行编译，支持多线程加速：
   ```sh
   cmake --build build --parallel
   ```

3. 安装编译产物

   执行安装命令，将编译生成的二进制文件整理到`build_out`文件夹下：
   ```sh
   cmake --install build --prefix ./build_out
   ```
