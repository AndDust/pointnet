import os
import argparse
import time


if __name__ == "__main__":
    # 指定要创建的文件名®Ω
    file_name = "record.txt"
    # 使用'w'模式打开文件，如果文件不存在则会创建它
    # 'w'表示写入模式，如果文件已存在，它会被清空
    record_path = "/home/nku524/dl/codebase/pointnet/" + file_name

    # with open(record_path, 'a'):
    #     pass  # 使用'with'语句打开文件后，可以执行一些操作，这里暂时不做任何操作
    # print(f"已成功创建文件 '{file_name}'")

    with open(record_path, 'a', encoding='utf-8') as file:
        # 写入数据到文件
        file.write("____________________________________" + "\n")

    num_samples_list = [32, 16, 8, 4, 1]
    w_list = [8, 4]
    a_list = [8, 4]
    for i in range(len(num_samples_list)):
        num_samples = num_samples_list[i]
        for j in range(len(w_list)):
            for k in range(len(a_list)):
                w = w_list[j]
                a = a_list[k]

                os.system(
                    f'python main_pointnet.py --model ~/dl/codebase/pointnet/utils/cls/cls_model_249.pth --n_bits_w {w} --n_bits_a {a} --setgpu 1 --record {record_path} --pointnet_num_samples {num_samples}')
                time.sleep(0.5)

    f'python main_pointnet.py --model ~/dl/codebase/pointnet/utils/cls/cls_model_249.pth --n_bits_w 8 --n_bits_a 8 --setgpu 1 --a_count 11'
    # parser = argparse.ArgumentParser()
    # parser.add_argument("exp_name", type=str, choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    # args = parser.parse_args()
    # w_bits = [8, 4]
    # a_bits = [8, 4]
    #
    # """
    #     将权重和激活分别量化为2、4bit和2、4bit的组合，也就是4种组合
    # """
    #
    #
    # if args.exp_name == "resnet18":
    #     for i in range(4):
    #         os.system(f"python main_imagenet.py --data_path /home/nku524/dl/dataset/imageNet-1k --arch resnet18 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02")
    #         time.sleep(0.5)
    #
    # if args.exp_name == "resnet50":
    #     for i in range(4):
    #         os.system(f"python main_imagenet.py --data_path /home/nku524/dl/dataset/imageNet-1k --arch resnet50 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02")
    #         time.sleep(0.5)
    #
    # if args.exp_name == "regnetx_600m":
    #     for i in range(4):
    #         os.system(f"python main_imagenet.py --data_path /home/nku524/dl/dataset/imageNet-1k --arch regnetx_600m --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01")
    #         time.sleep(0.5)
    #
    # if args.exp_name == "regnetx_3200m":
    #     for i in range(4):
    #         os.system(f"python main_imagenet.py --data_path /home/nku524/dl/dataset/imageNet-1 --arch regnetx_3200m --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01")
    #         time.sleep(0.5)
    #
    # if args.exp_name == "mobilenetv2":
    #     for i in range(4):
    #         os.system(f"python main_imagenet.py --data_path /home/nku524/dl/dataset/imageNet-1 --arch mobilenetv2 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.1 --T 1.0 --lamb_c 0.005")
    #         time.sleep(0.5)
    #
    # if args.exp_name == "mnasnet":
    #     for i in range(4):
    #         os.system(f"python main_imagenet.py --data_path /home/nku524/dl/dataset/imageNet-1 --arch mnasnet --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.2 --T 1.0 --lamb_c 0.001")
    #         time.sleep(0.5)
    #
    #