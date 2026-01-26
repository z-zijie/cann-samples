# CANN Samples

## 快速开始

1. 配置项目

   使用以下命令初始化构建配置，CMake会自动创建`build`文件夹
   ```sh
   cmake -S . -B build
   ```

2. 查看可用的Target(可选)

   在编译前，可查看当前项目中所有支持单独编译的目标列表
   ```sh
   cmake --build build --target help
   ```

3. 执行编译安装

   - 选项A: 编译指定的Target(部分构建)
   
     将`<target_name>` 替换为上一步查到的名称：
     ```sh
     cmake --build build --target <target_name>
     ```

   - 选项B: 编译所有Target(推荐，全量构建)
     
     支持多线程加速：
     ```sh
     cmake --build build --parallel
     ```
     
     安装编译产物：

     执行安装命令，将编译生成的二进制文件整理到`build_out`文件夹下：
     ```sh
     cmake --install build --prefix ./build_out
     ```
