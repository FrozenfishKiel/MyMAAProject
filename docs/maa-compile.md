# MaaFramework 编译部署（占位）

本仓库在 `maa-deps/maafw-5.2.6-win_amd64/` 内提供了可直接使用的运行时依赖（Windows x64）。

如需自行编译 MaaFramework（用于二次开发或自定义插件），建议遵循官方构建文档：

- Windows：Visual Studio 2022 + CMake
- Linux：GCC/Clang + CMake

如需自行编译 MaaFramework（用于二次开发或自定义插件），可将产物放入 `maa-deps/` 并在 `src/main.py` 中调整加载路径。
