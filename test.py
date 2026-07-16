from analyzer import *
m=AudioAnalyzer(model_path='models/best_model.pt')
print(m.predict('四月森林的色调是如此的绿色，看起来明亮而富有诗意。', 'test.wav'))
#print(m.predict('我每个除夕夜都睡觉', 'test.wav'))
# print(m.predict('的 林', 'test.wav'))
# print(m.predict('秀', 'test.wav'))
# print(m.predict('的 底然', 'test.wav'))