# mnist 数据集图片的长宽
SIZE = 28

# mnist 训练集大小
TRAIN_SIZE = 60000

# mnist 测试集大小（用不到）
TEST_SIZE = 10000

# 发现连通的至少多少个颜色块才会被认为是数字
OCR_THRESHOLD = 100

# 在对图片进行 resize 成 SIZE * SIZE 大小的时候，上下左右保留的空白边界大小
BORDER = 4

# KNN 中的超参数
K = 10

# 预测结果存储桶
S3prediction = 'mnist-prediction'

# boxes存储桶
S3boxes = 'mnist-boxes'

# 原图存储桶
S3original = 'mnist-input'

# 输出图片存储桶
S3output = 'mnist-output'