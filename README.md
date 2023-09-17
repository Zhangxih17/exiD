# exiD
exiD data process and simulation: 

## 数据结构
```
├── data    
│   ├── maps    
│   ├── tracks    
│   ├── train                     
│   │       └── 0    
│   │           ├── 00            
│   │           └── 01               
├── data_loader                  
│   ├── data_process.py                   
│   ├── dataset.py                 
│   ├── exiD_loader.py                    
├── model                  
│   ├── GAT.py              
│   ├── MTR.py                  
│   └── subNetwork.py                     
├── README.md                   
├── result              
│   └── 50b100e.png                
└── src                 
    ├── cf_IDM.ipynb                                      
    ├── main.py      
    ├── MTR.ipynb                
    ├── test.py                    
    └── VectorNet.ipynb                

 ```                     

## 数据处理
- `data_process` :  读取原始exiD数据集并生成用于训练csv (5s/50帧)
- `exiD_loader` :  参考 `ArgoForcastLoader` 读取类
- `dataset` :  打包标准训练集

                                             
 ## 模型
- `VectorNet` : colab 复现 
- `MTR` : MTR+GNN 模型复现 (20%)
                                  
                            
TODO:
- motion-query pair
- GMM output
- ...


