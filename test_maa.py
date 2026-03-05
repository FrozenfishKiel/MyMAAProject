import sys
sys.path.insert(0, 'maa-deps/maafw-5.2.6-win_amd64')

try:
    import maa
    print("MaaFramework 导入成功")
    print(f"有 __version__ 属性: {hasattr(maa, '__version__')}")
    if hasattr(maa, 'library'):
        print(f"有 library 模块: True")
        try:
            version = maa.library.Library.version()
            print(f"MaaFramework 版本: {version}")
        except Exception as e:
            print(f"获取版本失败: {e}")
except Exception as e:
    print(f"MaaFramework 导入失败: {e}")
    import traceback
    traceback.print_exc()
