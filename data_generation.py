import csv
import random
import time

x_value = 0
total_1 = 1000
total_2 = 1000
total_3 = 1000

fieldnames = ["x_value", "total_1", "total_2", "total_3"]

with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

while True:
    with open('data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            'x_value' : x_value,
            'total_1' : total_1,
            'total_2' : total_2,
            'total_3' : total_3,
        }

        csv_writer.writerow(info)
        print(x_value, total_1, total_2, total_3)

        x_value += 1
        total_1 += random.randint(0, 2)
        total_2 += random.randint(0, 2)
        total_3 += random.randint(0, 2)

    time.sleep(1)
