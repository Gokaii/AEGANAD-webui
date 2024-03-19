import gradio as gr
import webbrowser
import yaml
import torch
import os
import shutil
import csv
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
dev_root='/mnt/disk_4t/public/guokai/dcase2020/dev_data'
best_dict={
    'valve':'G_x_2_max',
    'fan':'D_lof',
    'pump':'D_lof',
    'slider':'G_x_2_min',
    'ToyCar':'D_maha',
    'ToyConveyor':'D_maha'
}
threshold = {
    'valve':1,
    'fan':1,
    'pump':1,
    'slider':1,
    'ToyCar':1,
    'ToyConveyor':1
}

def calc_specgram(aud):
    # sr=None声音保持原采样频率， mono=False声音保持原通道数
    data, fs = librosa.load(aud, sr=None, mono=False)
    #data = aud[1]
    L = len(data)
    print('Time:', L / fs)
    #0.025s
    framelength = 0.025
    #NFFT点数=0.025*fs
    framesize = int(framelength * fs)
    print("NFFT:", framesize)
    #画语谱图
    plt.specgram(data, NFFT=framesize, Fs=fs, window=np.hanning(M=framesize))
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')

    return plt

def calc_melspecgram(aud):
    data, fs = librosa.load(aud, sr=None, mono=False)
    L = len(data)
    print('Time:', L / fs)
    # 0.025s
    framelength = 0.025
    # NFFT点数=0.025*fs
    framesize = int(framelength * fs)
    print("NFFT:", framesize)
    #提取mel特征
    mel_spect = librosa.feature.melspectrogram(data, sr=fs, n_fft=framesize)
    #转化为log形式
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    print('plot: ',mel_spect.shape)
    #画mel谱图
    librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    return plt


def init_metrics():
    D_metric = ['D_maha', 'D_knn', 'D_lof', 'D_cos']
    G_metric = ['G_x_2_sum', 'G_x_2_min', 'G_x_2_max', 'G_x_1_sum', 'G_x_1_min', 'G_x_1_max',
                'G_z_2_sum', 'G_z_2_min', 'G_z_2_max', 'G_z_1_sum', 'G_z_1_min', 'G_z_1_max',
                'G_z_cos_sum', 'G_z_cos_min', 'G_z_cos_max']
    all_metric = D_metric + G_metric
    return all_metric

def clean_infer_path():
    for mtp in os.listdir(dev_root):
        shutil.rmtree(dev_root+'/'+mtp+'/infer')
        os.makedirs(dev_root+'/'+mtp+'/infer')
        


def start_train(card,mt,batch_size,epoch):
    config = load_cfg()
    config['train']['bs']=batch_size
    config['train']['epoch']=epoch
    with open('config.yaml','w',encoding='utf-8') as cfg:
        yaml.dump(config, cfg)

    cmd = 'python'+' '+'train.py'+' '+'--mt'+' '+mt+' '+'-c'+' '+card
    print(cmd)
    status = 'ok'
    return status

def start_infer(card,mt,aud):
    clean_infer_path()
    target_infer_path = dev_root+'/'+mt+'/infer/anomaly_id_00_00000000.'+ aud.split('.')[-1]
    cmd = 'mv'+' '+aud+' '+target_infer_path
    print(cmd)
    os.system(cmd)
    cmd = 'python'+' '+'infer.py'+' '+'--mt'+' '+mt+' '+'-c'+' '+card
    print(cmd)
    os.system(cmd)
    status = 'ok'
    clean_infer_path()
    
    infer_result=pd.read_csv('cache/infer_res.csv',index_col=False)
    #print(infer_result)
    infer_result=infer_result.T
    #print(infer_result)

    # mel_spect = np.load('cache/recon.npy')
    # librosa.display.specshow(mel_spect, sr=16000, x_axis='time', y_axis='mel')
    # plt.ylabel('Mel Frequency')
    # plt.xlabel('Time(s)')
    # plt.title('Mel Spectrogram')

    plt_re = recon_plt()
    #plt_sub = subtract_plt()

    best_metric = best_dict[mt]
    thres = threshold[mt]
    met = init_metrics()
    metric2id = {m: meid for m, meid in zip(met, range(len(met)))}
    best_metid = metric2id[best_metric]
    score = infer_result[1][best_metid]
    #print('inf',infer_result[1][best_metid])
    if float(score) > float(thres):
        decision = '异常'
    else:
        decision = '正常'
    info = '检测结果为：'+decision+'\n最佳指标为: '+ best_metric+'\n判断阈值为: '+ str(thres)+'\n异常分数为: '+ score
    return status, infer_result, plt_re, info

def recon_plt():
    print('111111111111111111111111111111')
    mel_spect = np.load('cache/recon.npy')
    
    librosa.display.specshow(mel_spect, sr=16000, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    return plt

def subtract_plt():
    recon = np.load('cache/recon.npy')
    inp = np.load('cache/in.npy')
    res = np.subtract(inp,recon)
    print('res', res)
    
    librosa.display.specshow(res, sr=16000, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    return plt



def load_cfg():
    with open('config.yaml',encoding='utf-8') as cfg:
        data = yaml.load(cfg,Loader=yaml.FullLoader)
    return data


if __name__ == '__main__':
    machines = ['fan','pump','slider','ToyCar','ToyConveyor','valve']
    cfg = load_cfg()
    clean_infer_path()
    current_bs = cfg['train']['bs']
    current_epo = cfg['train']['epoch']
    cards = [i for i in range(torch.cuda.device_count())]
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem('训练'):
                    with gr.Column():
                        card = gr.Dropdown(cards,label='使用显卡',value=cards[0])
                        mt = gr.Dropdown(machines,label='机器类型')
                        batch_size = gr.Slider(
                                label="批大小",
                                value=current_bs,
                                minimum=1,
                                maximum=512,
                                step=1,
                            )
                        epoch = gr.Slider(
                                label="训练轮数",
                                value=current_epo,
                                minimum=1,
                                maximum=100,
                                step=1,
                            )
                        start_train_btn=gr.Button('开始训练')
                        train_status=gr.Textbox(label='训练状态')
                start_train_btn.click(fn=start_train,
                                    inputs=[card,mt,batch_size,epoch], 
                                    outputs=[train_status])
                with gr.TabItem('推理'):
                    with gr.Column():
                        aud = gr.Audio(label='请上传待检测音频',type='filepath')
                        mel_plot = gr.Plot(label='梅尔谱图')
                        mt = gr.Dropdown(machines,label='机器类型')
                        card = gr.Dropdown(cards,label='使用显卡',value=cards[0])
                        start_infer_btn=gr.Button('开始检测')
                        infer_status=gr.Textbox(label='检测状态')
                        with gr.Row():
                            with gr.Column():
                                mel_plot_recon = gr.Plot(label='重构梅尔谱')
                            with gr.Column():
                                mel_plot_sub = gr.Plot(label='异常可视化')
                        with gr.Row():
                            with gr.Column():
                                infer_result=gr.DataFrame(label='异常分数',interactive=False,headers=['Metrics','Scores'])
                            with gr.Column():
                                decision=gr.Textbox(label='检测结果')
                aud.change(fn=calc_melspecgram,
                               inputs=[aud],
                               outputs=[mel_plot])        
                infer_result.change(fn=subtract_plt,
                                      inputs=[],
                                      outputs=[mel_plot_sub])

                start_infer_btn.click(fn=start_infer,
                                    inputs=[card,mt,aud], 
                                    outputs=[infer_status,infer_result,mel_plot_recon, decision])
                        
    webbrowser.open("http://127.0.0.1:6677")
    demo.launch(server_name='0.0.0.0', 
                server_port=6677,
                share=True)
