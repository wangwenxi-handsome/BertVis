import os
import torch 
import random
from transformers import BertTokenizer, BertConfig
from transformers import BertModel

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
# 解决中文不显示的问题, 如果本地可以显示中文，就不需要从本地加载中文字体
from matplotlib import font_manager
zh_font = font_manager.FontProperties(fname="SimHei.ttf")


class BertVis:
    def __init__(
        self, 
        model,
        tokenizer,
        file_path,
        include_labels = False,
        max_length = 10,
        save_path = "./",
        plot_nums = 1000,
    ):
        """
            model: bert model
            tokenizer: bert tokenizer
            file_path: 数据，如果数据中有label，则每一行为words/sentence + "\t" + label, 注意两者用"\t"分割；如果没有label则每一行为一句话
            include_labels: 数据中是否包含label
            max_length: model接受的句子最长长度，超过的会被截断
            save_path: PCA和TSNE图的储存路径
            plot_nums: 从所有数据中随机选择plot_nums个句子进行embedding可视化
        """
        self.model = model
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.save_path = save_path
        self.include_labels = include_labels
        self.words, self.labels = self.load_words(file_path, plot_nums, include_labels)
        self.embeddings = self.get_embedding(self.words, max_length)
        
    def load_words(self, file_path, plot_nums, include_labels):
        with open(file_path, "r") as f:
            data = f.readlines()
            
            if include_labels:
                # 如果数据中有标签
                data = [i.strip().split("\t") for i in data]
                data = random.sample(data, plot_nums)
                words = [i[0] for i in data]
                labels = [int(i[1]) for i in data]
                return words, labels
            else:
                # 如果数据中没有标签
                words = [i.strip() for i in data]
                words = random.sample(words, plot_nums)
                return words, None
        
    def get_embedding(self, words, max_length):
        # 从model中获取embedding
        inputs = self.tokenizer(words, padding = True, truncation=True, return_tensors="pt", max_length=max_length)
        with torch.no_grad():
            embeddings = self.model(**inputs).pooler_output.numpy()
        return embeddings
    
    def get_vis(self, mode = "PCA"):
        # 对embedding做降维处理，并保存图片
        if mode == "PCA":
            dr = PCA(n_components = 2)
        elif mode == "TSNE":
            dr = TSNE(n_components = 2)
        else:
            raise ValueError(f"{mode} is not supported")
        
        # pca or tsne
        results = dr.fit_transform(self.embeddings)
        
        # plot
        plt.cla()
        if not self.include_labels:
            plt.scatter(results[:, 0], results[:, 1], c = "b", s = 5, cmap = 'plasma', alpha = 0.6)
        else:
            # 只有8种颜色，所以会有相同类别为同一种颜色，这里用了一个小trick，用取余来决定该类别使用什么颜色
            all_color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
            point_color = [all_color[i % 8] for i in self.labels]
            plt.scatter(results[:, 0], results[:, 1], c = point_color, s = 5, cmap = 'plasma', alpha = 0.6)

        fig_name = self.file_path.split(".")[0]
        plt.savefig(os.path.join(self.save_path, f"{fig_name}_{mode}.jpg"))
    

if __name__ == "__main__":
    # load model and tokenizer
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_config = BertConfig.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, config = model_config)
    
    # vis
    # include label
    vis1 = BertVis(
        model, 
        tokenizer,
        file_path = "words_labels.txt",
        include_labels = True,
        plot_nums = 1000,
    )
    vis1.get_vis("PCA")
    vis1.get_vis("TSNE")
    
    # not include label
    vis2 = BertVis(
        model, 
        tokenizer,
        file_path = "words.txt",
        include_labels = False,
        plot_nums = 1000,
    )
    vis2.get_vis("PCA")
    vis2.get_vis("TSNE")