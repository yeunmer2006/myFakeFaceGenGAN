import torch

def check_gpu_availability():
    """检查GPU可用性及环境配置"""
    print("="*50)
    print("PyTorch GPU 可用性检查")
    print("="*50)
    
    # 1. 基本PyTorch信息
    print(f"\n[PyTorch版本] {torch.__version__}")
    print(f"[CUDA可用] {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n⚠️ 警告: PyTorch未检测到GPU支持！")
        return
    
    # 2. GPU设备信息
    device_count = torch.cuda.device_count()
    print(f"\n[GPU数量] {device_count}")
    
    for i in range(device_count):
        print(f"\n[GPU {i} 详细信息]")
        print(f"名称: {torch.cuda.get_device_name(i)}")
        print(f"计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"总显存: {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GB")
    
    # 3. CUDA信息
    print(f"\n[CUDA版本] {torch.version.cuda}")
    print(f"[cuDNN版本] {torch.backends.cudnn.version()}")
    
    # 4. 性能测试
    print("\n[性能测试]")
    try:
        x = torch.randn(10000, 10000).cuda()
        y = torch.randn(10000, 10000).cuda()
        _ = x @ y
        print("✅ GPU矩阵乘法测试成功")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU测试失败: {str(e)}")
    
    # 5. 环境检查
    print("\n[环境检查]")
    print("建议的PyTorch-GPU安装命令:")
    print(f"pip install torch=={torch.__version__} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{torch.version.cuda.replace('.','')[:2]}")

if __name__ == "__main__":
    check_gpu_availability()
    
    # 附加：显示当前设备使用情况
    print("\n[当前设备状态]")
    print(f"默认设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"当前分配显存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"最大保留显存: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
