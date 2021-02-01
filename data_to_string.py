import numpy as np
import sqlite3

np_array = np.array(["Check"])

conn = sqlite3.connect('scores_main_c.db')
print("Opened database successfully")

# cursor = conn.execute("SELECT ID, name, address, salary from COMPANY")
cursor = conn.execute(
    "SELECT ID, TWEET, USER_ID, US_FL, US_FL_N, TW_DATE, WORD_COUNT, TW_TAGS, TW_LIKES, TW_RT, MEDIA_YN, MEDIA_LINKS from JUST_TWEETS where MEDIA_YN = 0")

count = 0;

for row in cursor:
    print(count)
    count += 1
    np_array_temp = np.array([row[1]])
    # print(np_array_temp)
    np_array = np.append(np_array, np_array_temp)


# print("Operation done successfully")
conn.close()

print("\"" + str.format(np_array[2]) + "\",")

with open("output2.txt", "w") as txt_file:
    for row in np_array:
       txt_file.write("\"" + str.format(row) + "\",")  # works with any number of elements in a line
