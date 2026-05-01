import re

# 输入文本
text = "05/01 14:58:33 - mmengine - INFO - Epoch(test) [205/205]    empty: 0.7384  floor: 0.7004  wall: 0.5134  chair: 0.5193  cabinet: 0.2466  door: 0.2424  table: 0.4088  couch: 0.4474  shelf: 0.3895  window: 0.2574  bed: 0.4876  curtain: 0.4190  desk: 0.2576  doorframe: 0.2137  plant: 0.2529  stairs: 0.3510  pillow: 0.3510  wardrobe: 0.0603  picture: 0.1868  bathtub: 0.5990  box: 0.1287  counter: 0.2494  bench: 0.1727  stand: 0.1947  rail: 0.1466  sink: 0.3972  clothes: 0.1426  mirror: 0.1696  toilet: 0.5047  refrigerator: 0.1394  lamp: 0.2742  book: 0.0750  dresser: 0.0621  stool: 0.1073  fireplace: 0.0351  tv: 0.1427  blanket: 0.1409  commode: 0.0061  washing machine: 0.0891  monitor: 0.5024  window frame: 0.0031  radiator: 0.3290  mat: 0.0185  shower: 0.0068  rack: 0.0030  towel: 0.1847  ottoman: 0.1338  column: 0.0174  blinds: 0.0000  stove: 0.0950  bar: 0.0817  pillar: 0.0527  bin: 0.3080  heater: 0.0675  clothes dryer: 0.0064  backpack: 0.1859  blackboard: 0.1509  decoration: 0.0218  roof: 0.0000  bag: 0.0401  steps: 0.1173  windowsill: 0.0947  cushion: 0.0102  carpet: 0.0804  copier: 0.2555  board: 0.0211  countertop: 0.0119  basket: 0.0444  mailbox: 0.0000  kitchen island: 0.0132  washbasin: 0.0109  bicycle: 0.0000  drawer: 0.0075  oven: 0.0721  piano: 0.0455  excercise equipment: 0.0000  beam: 0.0000  partition: 0.0000  printer: 0.1472  microwave: 0.1398  frame: 0.0000  data_time: 1.8619  time: 2.1763"
# 使用正则表达式提取类别和数值
pattern = re.compile(r"(\w+|refrigerator|toilet|washing machine|clothes dryer|blackboard|windowsill|excercise equipment|window frame|kitchen island): (\d+\.\d+)")
matches = pattern.findall(text)

# 创建字典存储类别和数值
data = {item[0]: float(item[1]) for item in matches}
print(data)

# 指定需要打印的类别
categories = ["empty", "floor", "wall", "chair", "cabinet", "door", "table", "couch", "shelf", "window", "bed", "curtain", "refrigerator", "plant", "stairs", "toilet"]


categories_all = ['empty','floor', 'wall', 'chair', 'cabinet', 'door', 'table', 'couch',
               'shelf', 'window', 'bed', 'curtain', 'desk', 'doorframe',
               'plant', 'stairs', 'pillow', 'wardrobe', 'picture', 'bathtub',
               'box', 'counter', 'bench', 'stand', 'rail', 'sink', 'clothes',
               'mirror', 'toilet', 'refrigerator', 'lamp', 'book', 'dresser',
               'stool', 'fireplace', 'tv', 'blanket', 'commode',
               'washing machine', 'monitor', 'window frame', 'radiator', 'mat',
               'shower', 'rack', 'towel', 'ottoman', 'column', 'blinds',
               'stove', 'bar', 'pillar', 'bin', 'heater', 'clothes dryer',
               'backpack', 'blackboard', 'decoration', 'roof', 'bag', 'steps',
               'windowsill', 'cushion', 'carpet', 'copier', 'board',
               'countertop', 'basket', 'mailbox', 'kitchen island',
               'washbasin', 'bicycle', 'drawer', 'oven', 'piano',
               'excercise equipment', 'beam', 'partition', 'printer',
               'microwave', 'frame']


# 打印第一行类别名称
print(" ".join(categories))
# 打印第二行对应的数值
print(" ".join(str(data.get(category, None)) for category in categories))
# 打印第一行类别名称
print(" ".join(categories_all))
# 打印第二行对应的数值
print(" ".join(str(data.get(category, None)) for category in categories_all))